"""
Context directory builder for the single multi-turn FGA Claude Agent SDK session.

Provides:
  bfs_topological_sort    — order workflow nodes for BFS-first implementation
  setup_context_directory — build the COMPLETE context once, before the session:
                            skeleton + FaaSr stubs, shared CONTEXT.md, per-node
                            specs/<fn>.md, seeded functions/ and test_data/
"""
from __future__ import annotations
import os
import shutil
from collections import deque
from pathlib import Path

from faasr_agents.models import FunctionSpec, WorkflowSpec, WorkflowEdge, ContextFile

# Written verbatim to stubs/faasr_stubs.py inside each context directory.
_STUBS_CODE = '''\
from __future__ import annotations
import os
import shutil


def faasr_get_file(local_file: str, remote_folder: str, remote_file: str) -> None:
    # test_output/ holds files produced by earlier stub runs this session (an
    # upstream function's REAL output wins); test_data/ holds seeded inputs.
    for d in ("test_output", "test_data"):
        src = os.path.join(d, remote_file)
        if os.path.exists(src):
            shutil.copy(src, local_file)
            return
    open(local_file, "w").close()


def faasr_put_file(local_file: str, remote_folder: str, remote_file: str) -> None:
    os.makedirs("test_output", exist_ok=True)
    if os.path.exists(local_file):
        shutil.copy(local_file, os.path.join("test_output", remote_file))


def faasr_log(msg: str) -> None:
    print(f"[FaaSr] {msg}", flush=True)


def faasr_secret(name: str) -> str:
    return os.environ[name]


def faasr_rank() -> dict:
    return {
        "rank": int(os.environ.get("FAASR_RANK", "1")),
        "max_rank": int(os.environ.get("FAASR_MAX_RANK", "1")),
    }


def faasr_get_folder_list(server_name="", prefix=""):
    # Real FaaSr returns FULL S3 object keys (folder prefix included).
    names = set()
    for d in ("test_output", "test_data"):
        if os.path.isdir(d):
            names.update(os.listdir(d))
    if not prefix:
        return sorted(names)
    if "/" in prefix:
        dirpart, frag = prefix.rsplit("/", 1)
        return sorted(f"{dirpart}/{n}" for n in names if n.startswith(frag))
    return sorted(f"{prefix}/{n}" for n in names)
'''


# ---------------------------------------------------------------------------
# BFS topological sort
# ---------------------------------------------------------------------------

def bfs_topological_sort(
    nodes: list[FunctionSpec],
    edges: list[WorkflowEdge],
) -> list[FunctionSpec]:
    """
    Kahn's-algorithm topological sort on the workflow DAG.

    Nodes with no incoming edges come first (BFS level 0), followed by
    their successors, and so on.  Disconnected or cycle nodes are appended
    at the end in their original order.
    """
    name_to_node: dict[str, FunctionSpec] = {n.name: n for n in nodes}
    graph: dict[str, list[str]] = {n.name: [] for n in nodes}
    in_degree: dict[str, int] = {n.name: 0 for n in nodes}

    for edge in edges:
        if edge.from_node in graph and edge.to_node in graph:
            graph[edge.from_node].append(edge.to_node)
            in_degree[edge.to_node] += 1

    queue: deque[str] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    ordered: list[FunctionSpec] = []
    while queue:
        name = queue.popleft()
        ordered.append(name_to_node[name])
        for successor in graph[name]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    ordered_names = {n.name for n in ordered}
    ordered.extend(n for n in nodes if n.name not in ordered_names)
    return ordered


# ---------------------------------------------------------------------------
# Directory setup (once per fga_node invocation — the session's whole world)
# ---------------------------------------------------------------------------

def setup_context_directory(
    context_dir: Path,
    workflow: WorkflowSpec,
    context_files: list[ContextFile] | None = None,
    *,
    generate_nodes: list[FunctionSpec] | None = None,
    feedback: str = "",
    user_request: str = "",
) -> None:
    """
    Build the complete context directory for the single FGA session:

    - skeleton subdirs + FaaSr stubs
    - user-attached reference files copied into context/
    - functions/ seeded with every node that already has code: catalog/cached
      reuse and adapt/feedback seeds as functions/<name>.py; user-provided
      originals as functions/original_<name>.py (parity baseline — the agent
      writes the FaaSr translation to functions/<name>.py fresh)
    - test_data/ seeded with user-provided real inputs for EVERY node's declared
      inputs (upstream outputs are picked up at stub-run time from test_output/,
      so no per-node reseeding is needed)
    - CONTEXT.md — the shared, node-invariant workflow context (read once)
    - specs/<fn>.md — the per-node specification for each node in generate_nodes
    """
    for subdir in ("context", "stubs", "test_data", "test_output", "functions", "specs"):
        (context_dir / subdir).mkdir(exist_ok=True)

    (context_dir / "stubs" / "faasr_stubs.py").write_text(_STUBS_CODE)
    (context_dir / "stubs" / "__init__.py").write_text(
        "from faasr_stubs import faasr_get_file, faasr_put_file, faasr_log, "
        "faasr_secret, faasr_rank, faasr_get_folder_list\n"
    )

    ctx = context_dir / "context"
    for cf in context_files or []:
        try:
            shutil.copy2(cf.path, ctx / cf.name)
        except OSError as ex:
            print(f"  ⚠  FGA  skipping reference file {cf.name!r} — {ex}", flush=True)

    fns = context_dir / "functions"
    for node in workflow.nodes:
        if not node.code:
            continue
        if node.source == "user_provided":
            (fns / f"original_{node.name}.py").write_text(node.code)
        else:
            (fns / f"{node.name}.py").write_text(node.code)

    for node in workflow.nodes:
        _write_test_data(context_dir / "test_data", node, context_files)

    (context_dir / "CONTEXT.md").write_text(
        _shared_context_md(workflow, context_files, user_request)
    )

    specs = context_dir / "specs"
    for node in generate_nodes or []:
        (specs / f"{node.name}.md").write_text(
            _node_spec_md(node, workflow, feedback, context_files)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LARGE_CSV_BYTES = 512 * 1024  # truncate matched CSVs above this so stub runs stay fast


def _seed_real_input(dst: Path, src_path: str) -> bool:
    """Copy a user-provided context file into test_data/ as a real test input.

    Large CSVs are truncated (header + first 500 data rows) so a local stub run stays
    fast while remaining schema-complete; everything else is copied verbatim. Returns
    True on success, False if the source can't be read (caller falls back to a placeholder).
    """
    import itertools
    try:
        if dst.suffix.lower() == ".csv" and os.path.getsize(src_path) > _LARGE_CSV_BYTES:
            with open(src_path, "r", errors="replace") as f:
                head = list(itertools.islice(f, 501))  # header + up to 500 data rows
            dst.write_text("".join(head))
        else:
            shutil.copy2(src_path, dst)
        return True
    except OSError:
        return False


def _write_test_data(
    test_data_dir: Path,
    spec: FunctionSpec,
    context_files: list[ContextFile] | None = None,
) -> None:
    """Seed test_data/ with REAL user-provided inputs for this function's stub run.

    Upstream outputs are NOT copied here: the stub's faasr_get_file falls back to
    test_output/, where earlier functions' stub runs deposit their real outputs.
    If no user file matches, we intentionally write NOTHING — no fabricated
    placeholder. The FGA must then build a representative input (see CONTEXT.md →
    Testing with Stubs). We never invent schema-less data that would either
    silently false-pass or false-fail the function's own validation.
    """
    by_name: dict[str, str] = {cf.name: cf.path for cf in (context_files or [])}
    for inp in spec.inputs:
        # Ranked inputs carry a {rank} placeholder; materialize the rank=1 shard so
        # a local stub run (FAASR_RANK defaults to 1) finds a concrete file.
        fname = inp.name.format(rank=1) if "{rank}" in inp.name else inp.name
        path = test_data_dir / fname
        if path.exists():
            continue
        real = by_name.get(inp.name) or by_name.get(fname)
        if real:
            _seed_real_input(path, real)


def signature_and_test_call(spec: FunctionSpec) -> tuple[str, str]:
    sig_parts = (
        ["folder: str"]
        + [f"input{i + 1}: str" for i in range(len(spec.inputs))]
        + [f"output{i + 1}: str" for i in range(len(spec.outputs))]
    )
    signature = f"def {spec.name}({', '.join(sig_parts)}) -> None"

    test_args = (
        ["'test_folder'"]
        + [f"'{inp.name}'" for inp in spec.inputs]
        + [f"'{out.name}'" for out in spec.outputs]
    )
    test_call = f"{spec.name}({', '.join(test_args)})"
    return signature, test_call


def _shared_context_md(
    workflow: WorkflowSpec,
    context_files: list[ContextFile] | None = None,
    user_request: str = "",
) -> str:
    """The node-invariant workflow context, read ONCE at the start of the session."""
    all_fns = ", ".join(n.name for n in workflow.nodes)

    # The verbatim user request is the ground truth for implementation details that
    # upstream spec summarization may have dropped — always surface it to the agent.
    user_request_block = ""
    if (user_request or "").strip():
        user_request_block = (
            f"\n## Original User Request (verbatim)\n{user_request.strip()}\n\n"
            "Follow any explicit implementation details above that apply to the step "
            "you are implementing exactly as written — never substitute your own "
            "defaults for something the user specified. Details about other steps "
            "belong to those steps.\n"
        )

    user_context_block = ""
    if context_files:
        listing = "\n".join(
            f"  - context/{cf.name}" + (f" — {cf.description}" if cf.description else "")
            for cf in context_files
        )
        user_context_block = (
            f"\n## [User Provided Context] — Reference Files\n"
            f"The user attached these files as domain reference material (papers, "
            f"input/output examples, sample datasets). They live in context/ alongside "
            f"the code. READ the relevant ones to match real data formats, "
            f"parameter values, and domain details — but they are REFERENCES, not code "
            f"to import or execute, and not a license to fabricate data:\n"
            f"{listing}\n"
            f"Files whose names match a function's declared inputs have already been "
            f"placed in test_data/ as the REAL local test inputs. Stub runs MUST use "
            f"them and pass — do not test against placeholder data.\n"
        )

    return f"""\
# Workflow: {workflow.name}
{workflow.description}

## All Workflow Functions
{all_fns}

You implement them one per turn, in the order instructed. Each function's full
specification is in specs/<function>.md.
{user_request_block}
## Data Integrity (non-negotiable)
This is a REAL scientific workflow. Use the real data source / API named in each spec.
NEVER fabricate, mock, randomize, hardcode, or fall back to synthetic data when a
source is unavailable — if it cannot be reached, faasr_log the error and raise.

## Neighbor Code (functions/)
functions/ holds the REAL code of every function that already exists: catalog/cached
reuse, seeds, user-provided originals (original_<name>.py), and the implementations
you complete during this session. MATCH filenames against it: a function's input
filenames MUST equal the exact remote_file each producer passes to faasr_put_file,
and its output filenames MUST equal what each consumer passes to faasr_get_file.
The Inputs/Outputs in each spec are starting names — reconcile to the real code when
they differ. CATALOG functions are FIXED (their filenames cannot change), so conform
to a catalog neighbor's names exactly.

## FaaSr API
```python
faasr_get_file(local_file="local.csv", remote_folder=folder, remote_file=input1)
faasr_put_file(local_file="local.csv", remote_folder=folder, remote_file=output1)
faasr_log("Processing complete")
token = faasr_secret("SOME_API_TOKEN")   # credentials/tokens — see each spec's Required Secrets
r = faasr_rank()                          # parallel rank: {{"rank": i, "max_rank": N}} (ranked fns only)
names = faasr_get_folder_list(prefix=folder)   # FULL keys incl. folder prefix; strip before get: n.rsplit("/", 1)[-1]
```
These names are provided by the FaaSr deployment runtime. The final file must contain
NO faasr import of any kind — no `import faasr`, no `from faasr import ...`, no
`from FaaSr_py import ...`, no `faasr_stubs`, and no try/except ImportError guard
around such imports. Call the API functions as bare names, exactly as shown above.

## Testing with Stubs
The stubs run a function locally against the files in test_data/. Run it with the
exact test call given in the function's spec:
```python
import sys; sys.path.insert(0, "stubs")
from faasr_stubs import faasr_get_file, faasr_put_file, faasr_log, faasr_secret, faasr_rank, faasr_get_folder_list
exec(open("functions/<fn>.py").read())
<test call from specs/<fn>.md>
# Then inspect test_output/ for the produced files.
```
The stub faasr_get_file reads test_output/ first (the REAL outputs earlier functions
produced this session), then test_data/ (seeded user inputs) — so downstream functions
are automatically tested on their upstream functions' actual outputs.

YOU ARE NOT DONE UNTIL THIS RUNS CLEANLY on representative inputs:
- If this is a PURE-LOCAL function (parse / validate / transform / simulate on the given
  inputs / plot) it has NO external dependency, so the stub run MUST complete without error.
  A parse/validation/logic error is a REAL DEFECT — fix the code and re-run. Do NOT rationalize
  a failure as "expected placeholder data."
- test_data/ is pre-seeded with REAL inputs where available (user-provided files), and earlier
  functions' actual outputs are in test_output/. If a matching file is present, you MUST run
  against it and get a clean pass — never swap in dummy data.
- If an input file is ABSENT from both (no user file, no upstream output yet), construct a
  representative input (matching the documented schema / a context reference file) in test_data/,
  then verify a clean run. Testing on representative data is required; fabricating data INSIDE the
  function is still forbidden (see Data Integrity).
- The ONLY acceptable local failure is a genuine EXTERNAL fetch (API / URL / database) that needs
  network or credentials unavailable here — and only that. Everything else must pass.
{user_context_block}"""


def _node_spec_md(
    spec: FunctionSpec,
    workflow: WorkflowSpec,
    feedback: str,
    context_files: list[ContextFile] | None = None,
) -> str:
    """The per-node specification, written to specs/<fn>.md."""
    signature, test_call = signature_and_test_call(spec)

    inputs_lines = "\n".join(
        f'  input{i + 1} = "{inp.name}"  # type: {inp.type or "any"}'
        + (f" — {inp.description}" if inp.description else "")
        for i, inp in enumerate(spec.inputs)
    ) or "  (none)"

    outputs_lines = "\n".join(
        f'  output{i + 1} = "{out.name}"  # type: {out.type or "any"}'
        + (f" — {out.description}" if out.description else "")
        for i, out in enumerate(spec.outputs)
    ) or "  (none)"

    incoming = [df for df in workflow.data_flow if df.to_node == spec.name]
    outgoing = [df for df in workflow.data_flow if df.from_node == spec.name]

    incoming_lines = "\n".join(
        f"  '{df.file}' ({df.format or 'any'}) ← from {df.from_node}"
        for df in incoming
    ) or "  (none — this is a source node)"

    outgoing_lines = "\n".join(
        f"  '{df.file}' ({df.format or 'any'}) → to {df.to_node}"
        for df in outgoing
    ) or "  (none — this is a sink node, write final outputs)"

    deps = ", ".join(spec.dependencies) if spec.dependencies else "none"
    secrets_block = "\n".join(
        f'  - {s}   →   read with faasr_secret("{s}")' for s in spec.secrets
    ) or "  (none)"

    rank_by_name = {n.name: getattr(n, "rank", 1) or 1 for n in workflow.nodes}
    succ = [e.to_node for e in workflow.edges if e.from_node == spec.name]
    pred = [e.from_node for e in workflow.edges if e.to_node == spec.name]
    rank_block = ""
    if spec.rank > 1:
        rank_block += (
            f"\n## Parallel Execution (RANK = {spec.rank})\n"
            f"This function runs as {spec.rank} parallel instances. Call "
            "`r = faasr_rank()` (returns {{'rank': i, 'max_rank': N}}) and use r['rank'] "
            f"(1..{spec.rank}) to process ONLY this instance's shard. The input/output "
            "names above contain a {rank} placeholder — substitute r['rank'] before "
            "calling faasr_get_file / faasr_put_file (e.g. input1.format(rank=r['rank'])).\n"
        )
    for s in succ:
        n = rank_by_name.get(s, 1)
        if n > 1:
            rank_block += (
                f"\n## Fan-out to ranked successor `{s}` (×{n})\n"
                f"`{s}` runs as {n} parallel instances, so you MUST write exactly {n} output "
                f"shards — one per instance. THIS function is NOT ranked: do NOT call faasr_rank() "
                f"to get this count (faasr_rank() reports YOUR own rank, which is 1 here). "
                f"Hardcode the count as {n} and substitute {{rank}} with 1..{n}, e.g.\n"
                f"    for i in range(1, {n + 1}):\n"
                f"        faasr_put_file(local_file=..., remote_folder=folder, "
                f"remote_file=output1.replace(\"{{rank}}\", str(i)))\n"
            )
    for p in pred:
        n = rank_by_name.get(p, 1)
        if n > 1:
            rank_block += (
                f"\n## Fan-in from ranked predecessor `{p}` (×{n})\n"
                f"You run ONCE after all {n} instances of `{p}` finish. Read ALL of its "
                "per-rank outputs: discover them with faasr_get_folder_list(prefix=...) "
                "and faasr_get_file each one (do not assume a fixed count or use boto3). "
                "faasr_get_folder_list returns FULL object keys including the folder "
                "prefix; pass only the basename to faasr_get_file: "
                "remote_file=name.rsplit(\"/\", 1)[-1].\n"
            )

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"\n## Feedback from Code Review\n{feedback}\n"
            "\nAddress all points above in your implementation.\n"
        )

    is_user_provided = spec.source == "user_provided"

    prior_block = ""
    if spec.code and not is_user_provided:
        prior_block = (
            f"\n## Previous Implementation\n"
            f"See functions/{spec.name}.py — make the smallest change that addresses "
            "the feedback while preserving everything else.\n"
        )

    parity_block = ""
    if is_user_provided and spec.user_model_mode == "verbatim":
        parity_block = (
            f"\n## [User Provided Model] — Wrap Verbatim (REQUIRED)\n"
            f"`functions/original_{spec.name}.py` is a user-provided Python **script or "
            f"function**. The user's model code itself must NOT change — your job is only to "
            f"wrap it for FaaSr.\n"
            f"1. Read it and work out WHAT it computes — its real inputs (files it reads, CLI/"
            f"argparse args, hardcoded paths, stdin) and outputs (files written, stdout, return "
            f"value).\n"
            f"2. Copy the model's code into `functions/{spec.name}.py` UNCHANGED — no renaming, "
            f"no algorithm/constant/output-format changes. (If it is a plain script, you may "
            f"enclose its top-level logic in a function so it is callable, but every statement's "
            f"logic must be preserved.) Then add the FaaSr entry function `{spec.name}` with the "
            f"signature above: download inputs via faasr_get_file, convert formats as needed, "
            f"CALL the model, and upload its results via faasr_put_file, matching the declared "
            f"I/O filenames.\n"
            f"3. The entry function must produce its results ONLY by calling the preserved model "
            f"function(s) — never by reimplementing or duplicating any model logic in the wrapper "
            f"body.\n"
            f"4. Install the original's third-party imports (pip install …) so it is runnable.\n"
            f"5. VERIFY, exercising the WRAPPED entry function (not the model directly):\n"
            f"   (a) PARITY — run the original (`python functions/original_{spec.name}.py` with "
            f"whatever argv/input files/cwd it expects) on representative synthetic inputs to "
            f"capture baseline outputs; run the entry function on the equivalent inputs via the "
            f"stubs (see Testing with Stubs); compare the outputs.\n"
            f"   (b) INVOCATION — prove the entry function actually executes the model: during "
            f"the test, temporarily instrument the model function (e.g. a call counter or "
            f"sentinel), assert it fired when the entry function ran, then REMOVE the "
            f"instrumentation so the shipped code is untouched.\n"
            f"6. Record both checks in `test_output/parity_report.txt`. If the outputs cannot be "
            f"made to match without changing the model, faasr_log the discrepancy and FAIL — "
            f"never fabricate data and never modify the model to force a pass.\n"
        )
    elif is_user_provided:
        parity_block = (
            f"\n## [User Provided Model] — Refactor & Verify Parity (REQUIRED)\n"
            f"`functions/original_{spec.name}.py` is a user-provided Python **script or "
            f"function** (it may be a plain script: top-level statements, argparse, hardcoded "
            f"paths, an `if __name__ == \"__main__\"` block — NOT necessarily a clean function).\n"
            f"1. Read it and work out WHAT it computes — its real inputs (files it reads, CLI/"
            f"argparse args, hardcoded paths, stdin) and outputs (files written, stdout, return "
            f"value).\n"
            f"2. Refactor that logic into the FaaSr function `functions/{spec.name}.py` with the "
            f"signature above, reading inputs via faasr_get_file and writing outputs via "
            f"faasr_put_file, matching the declared I/O filenames. Apply EXACTLY the changes the "
            f"workflow requirements and user feedback call for (see Requirements/Feedback above) "
            f"— everything else must preserve the original computation: do not change untouched "
            f"algorithms, constants, or output formats.\n"
            f"3. Install the original's third-party imports (pip install …) so it is runnable.\n"
            f"4. VERIFY PARITY: construct representative synthetic inputs; RUN the original "
            f"(`python functions/original_{spec.name}.py` with whatever argv/input files/cwd it "
            f"expects) to capture its baseline outputs; run your FaaSr translation on the "
            f"equivalent inputs via the stubs (see Testing with Stubs); compare the outputs.\n"
            f"5. Iterate until the outputs match apart from the required changes, then write a "
            f"short `test_output/parity_report.txt` noting what you compared, that they agree, "
            f"and which deviations are intentional (tied to the requested changes). If they "
            f"cannot be made to match, faasr_log the discrepancy and FAIL — never fabricate data "
            f"to force a pass.\n"
        )

    seeded_note = ""
    if context_files:
        input_names = {
            (inp.name.format(rank=1) if "{rank}" in inp.name else inp.name)
            for inp in spec.inputs
        } | {inp.name for inp in spec.inputs}
        seeded = [cf.name for cf in context_files if cf.name in input_names]
        if seeded:
            seeded_note = (
                f"\n## Real Test Inputs (user-provided)\n"
                f"These user files match this function's declared inputs and have been placed "
                f"in test_data/ as the REAL local test inputs: {', '.join(seeded)}. Your stub "
                f"run MUST use them and pass (see CONTEXT.md → Testing with Stubs) — do not "
                f"test against placeholder data.\n"
            )

    return f"""\
# Function: `{spec.name}`
{spec.description}

## Function Signature
```python
{signature}
```

## Inputs  (S3 filenames — pass as `remote_file` to faasr_get_file)
{inputs_lines}

## Outputs  (S3 filenames — pass as `remote_file` to faasr_put_file)
{outputs_lines}
{rank_block}
## PyPI Dependencies (third-party packages only — NOT faasr, NOT stdlib)
{deps}

## Required Secrets (credentials / API tokens)
Read each one with faasr_secret("NAME") — never hardcode, and raise if missing.
On GitHub Actions these come from repository secrets; locally from env vars.
If you discover this function needs a credential not listed here, introduce it:
call faasr_secret("NEW_NAME") with an UPPER_SNAKE name — it is detected automatically
and the user is prompted to add it before deploy.
{secrets_block}

## Data Flow  (MATCH these filenames to the real code in functions/)
### Incoming files (produced by upstream functions):
{incoming_lines}

### Outgoing files (consumed by downstream functions):
{outgoing_lines}

## Local Stub Test Call
```python
{test_call}
```
{seeded_note}{parity_block}{feedback_block}{prior_block}"""
