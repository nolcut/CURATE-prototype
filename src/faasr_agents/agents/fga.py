from __future__ import annotations
import asyncio
import contextlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

from faasr_agents.llm import get_default_model, get_llm, using_anthropic, using_openai
from faasr_agents.pricing import record_sdk_usage, record_usage
from faasr_agents.models import FunctionSpec, clean_dependencies
from faasr_agents.state import AgentState
from faasr_agents.temp_paths import usable_temp_dir
from faasr_agents.faasr.context_dir import (
    bfs_topological_sort,
    setup_context_directory,
    signature_and_test_call,
)


def _build_sdk_env() -> dict[str, str]:
    """Build env-var overrides for the Agent SDK subprocess.

    Defaults to Bedrock (CLAUDE_CODE_USE_BEDROCK=1 + AWS/Bedrock credentials).
    Passes ANTHROPIC_API_KEY instead ONLY when the CLI opted in via --anthropic-api.
    """
    if using_anthropic():
        env: dict[str, str] = {}
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        return env

    env = {"CLAUDE_CODE_USE_BEDROCK": "1"}
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_BEARER_TOKEN_BEDROCK",
        "BEDROCK_API_KEY",
    ):
        val = os.environ.get(key)
        if val:
            env[key] = val
    # The Claude Code runtime authenticates to Bedrock via the STANDARD
    # AWS_BEARER_TOKEN_BEDROCK env var, not our custom BEDROCK_API_KEY name. Translate
    # it here so the SDK gets a usable Bedrock API key regardless of call order (don't
    # rely on _make_bedrock_llm having run first to populate AWS_BEARER_TOKEN_BEDROCK).
    if env.get("BEDROCK_API_KEY") and not env.get("AWS_BEARER_TOKEN_BEDROCK"):
        env["AWS_BEARER_TOKEN_BEDROCK"] = env["BEDROCK_API_KEY"]
    return env


def _debug_enabled() -> bool:
    """True when the CLI was launched with --debug (sets FAASR_DEBUG=1)."""
    return bool(os.environ.get("FAASR_DEBUG"))


# Node-invariant rules, appended to the session system prompt ONCE. The per-turn
# prompts stay small (function name + signature + test call + node-specific rules);
# everything here is sent a single time and survives any mid-session compaction.
_SHARED_RULES = """\
You are the FaaSr Function Generation Agent. Across this session you implement the
serverless Python functions of ONE scientific workflow, one function per user turn.

Your working directory is the workflow context directory; every path below is
relative to it (CONTEXT.md, specs/, context/, stubs/, functions/, test_data/,
test_output/). Do NOT prefix paths with /root/repo or any other base — read
`CONTEXT.md`, not `/root/repo/CONTEXT.md`.

DATA INTEGRITY (non-negotiable): these are REAL scientific workflows. NEVER generate
synthetic, mock, random, placeholder, or example data. NEVER add a try/except or any
fallback that fabricates, simulates, or hardcodes data when an API, URL, dataset,
file, credential, or network resource is missing or fails. If a required data source
cannot be reached, the function MUST faasr_log the error and raise an exception —
failing loudly is correct; inventing data is never acceptable.

For EVERY function `<fn>` you are asked to implement:
1. Read its spec in specs/<fn>.md (inputs, outputs, data flow, secrets, deps). Follow
   any explicit implementation details in CONTEXT.md's 'Original User Request' section
   that apply to that step exactly as written — never substitute your own defaults for
   something the user specified.
2. MATCH filenames to the real neighbor code in functions/*.py: your input filenames
   MUST equal the exact remote_file each producer passes to faasr_put_file, and your
   output filenames MUST equal what each consumer passes to faasr_get_file. The
   Inputs/Outputs in the spec are starting names — reconcile to the real code when
   they differ. CATALOG functions are FIXED (their filenames cannot change), so
   conform to a catalog neighbor's names exactly. context/ may also hold non-.py
   [User Provided Context] reference files (papers, input/output examples, sample
   datasets) — read the relevant ones to match real data formats and domain details;
   they are references, not code to import, and never a license to fabricate data.
3. Write the implementation to functions/<fn>.py:
   - Use only the FaaSr API: faasr_get_file / faasr_put_file / faasr_log /
     faasr_secret / faasr_rank / faasr_get_folder_list (and faasr_delete_file).
     These names are provided by the deployment runtime — the final file must contain
     NO faasr import of any kind: no `import faasr`, no `from faasr import ...`, no
     `from FaaSr_py import ...`, no faasr_stubs, and no try/except ImportError guard
     around such imports. Call them as bare names.
   - NEVER use boto3 or any direct/raw S3 access. To discover files in a folder
     (e.g. to read every output of a ranked predecessor), call
     faasr_get_folder_list(prefix=...) — it returns FULL object keys including the
     folder prefix (e.g. 'myfolder/part_1.json') — then faasr_get_file each one,
     passing only the basename: remote_file=name.rsplit("/", 1)[-1].
   - Process data via local temp files; upload to S3 only at the end.
   - Use the REAL data source/API named in the spec. No fabricated or fallback data
     (see DATA INTEGRITY above).
   - Credentials / API tokens: read each with faasr_secret("NAME"). The names under
     "Required Secrets" in the spec are a starting point — if you discover the
     function needs a credential that isn't listed (e.g. the API rejects requests
     without one), INTRODUCE it: call faasr_secret("NEW_NAME") with an
     UPPER_SNAKE_CASE name (^[A-Z][A-Z_0-9]*$). It is detected automatically and the
     user is told to add it before deploy. Never hardcode a credential and never
     fabricate data to avoid needing one; a missing secret makes faasr_secret raise —
     let it.
4. RUN it through the stubs against the files in test_data/, using the exact test
   call given in the turn instructions / spec:
   python3 -c "
   import sys; sys.path.insert(0, 'stubs')
   from faasr_stubs import faasr_get_file, faasr_put_file, faasr_log, faasr_secret, faasr_rank, faasr_get_folder_list
   exec(open('functions/<fn>.py').read())
   <test call>
   "
   The stub faasr_get_file also reads test_output/ — the REAL outputs the functions
   you implemented earlier this session produced — so downstream functions are tested
   on their upstream functions' actual outputs.
5. YOU ARE NOT DONE until this runs CLEANLY on representative inputs. If this is a
   PURE-LOCAL function (parse/validate/transform/simulate on the given inputs/plot)
   it has NO external dependency, so the run MUST succeed — a parse/validation/logic
   error is a REAL DEFECT to fix, NOT an 'expected placeholder failure.' If real
   inputs were provided (user files in test_data/), you MUST run on them and pass. If
   an input file is ABSENT from both test_data/ and test_output/, first build a
   representative input (matching the documented schema / a context reference file)
   in test_data/, then verify a clean run. (Building test inputs is fine; fabricating
   data INSIDE the function is still forbidden — see DATA INTEGRITY.)
6. The ONLY acceptable local failure is a genuine EXTERNAL fetch (API/URL/database)
   that needs network or credentials unavailable here — and only that. Everything
   else must pass; fix and repeat until it does.
7. Write the third-party PyPI packages your code imports to functions/<fn>.deps.txt,
   one package per line (the installable PyPI name — map the import to its package
   yourself, e.g. the name you would `pip install`). EXCLUDE the standard library,
   `faasr`, and `boto3`. The function runs in a minimal headless Linux container with
   no display server or GUI/system libraries — when a package offers a headless or
   slim variant, pick it over the full build (e.g. opencv-python-headless, never
   opencv-python). Write an empty file if there are none.

Do not modify a previously completed function unless the current turn's spec or
feedback explicitly requires it.
"""


_OPENAI_FGA_SYSTEM = """\
You are the FaaSr Function Generation Agent. Generate one production-quality Python
function for a scientific FaaSr workflow step.

Return ONLY a JSON object with this shape:
{
  "code": "complete Python source for functions/<fn>.py",
  "dependencies": ["installable-pypi-package", "..."]
}

The code must define the requested function with the exact signature. It must call
FaaSr runtime helpers as bare names: faasr_get_file, faasr_put_file, faasr_log,
faasr_secret, faasr_rank, and faasr_get_folder_list. Do not import faasr,
FaaSr_py, or faasr_stubs. Do not use boto3 or raw S3 access. Never fabricate,
mock, randomize, or hardcode scientific data; if a required external source,
file, credential, or network resource is unavailable, log and raise.
"""


def _extract_json_object(text) -> dict:
    """Extract a JSON object from an LLM response."""
    if isinstance(text, list):
        text = "\n".join(
            part.get("text", "")
            for part in text
            if isinstance(part, dict) and part.get("type") in ("text", "output_text")
        )
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group() if match else raw)


def _source_snapshot(context_dir: Path, current_name: str) -> str:
    """Compact code snapshot for neighbors already present in functions/."""
    parts: list[str] = []
    for path in sorted((context_dir / "functions").glob("*.py")):
        if path.name == f"{current_name}.py":
            continue
        try:
            code = path.read_text()
        except OSError:
            continue
        if len(code) > 12_000:
            code = code[:12_000] + "\n# ... truncated ..."
        parts.append(f"--- functions/{path.name} ---\n{code}")
    return "\n\n".join(parts) if parts else "(none)"


def _openai_turn_prompt(context_dir: Path, spec: FunctionSpec, feedback: str) -> str:
    """Build the OpenAI FGA prompt for one function."""
    signature, test_call = signature_and_test_call(spec)
    shared = (context_dir / "CONTEXT.md").read_text()
    node_spec = (context_dir / "specs" / f"{spec.name}.md").read_text()
    feedback_block = f"\n\nPrevious local test failure to fix:\n{feedback}" if feedback else ""
    return f"""\
Shared workflow context:
{shared}

Function specification:
{node_spec}

Existing neighboring function code:
{_source_snapshot(context_dir, spec.name)}

Required exact signature:
{signature}

Local stub test call that should run:
{test_call}
{feedback_block}

Generate the complete implementation for functions/{spec.name}.py and list only the
third-party PyPI dependencies it imports. Return JSON only.
"""


def _run_stub_test(context_dir: Path, spec: FunctionSpec, timeout: int = 60) -> tuple[bool, str]:
    """Run the generated function through the local FaaSr stubs."""
    _, test_call = signature_and_test_call(spec)
    script = (
        "import sys\n"
        "sys.path.insert(0, 'stubs')\n"
        "from faasr_stubs import faasr_get_file, faasr_put_file, faasr_log, "
        "faasr_secret, faasr_rank, faasr_get_folder_list\n"
        f"exec(open('functions/{spec.name}.py').read())\n"
        f"{test_call}\n"
    )
    try:
        proc = subprocess.run(
            ["python3", "-c", script],
            cwd=context_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"Stub test timed out after {timeout}s\n{exc}"
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return proc.returncode == 0, output[-6000:]


def _module_missing_due_to_reported_dep(test_output: str, deps: list[str]) -> bool:
    """Treat missing optional generated deps as install-time, not generation-time."""
    return bool(deps) and "ModuleNotFoundError" in test_output


def _implement_all_openai(
    ordered_nodes: list[FunctionSpec],
    context_dir: Path,
    model: str,
) -> list[FunctionSpec]:
    """Implement workflow nodes with the selected OpenAI chat model."""
    implemented: list[FunctionSpec] = []
    n = len(ordered_nodes)
    llm = get_llm(model)

    for idx, node in enumerate(ordered_nodes, 1):
        if node.source in ("catalog", "cached") and node.code:
            where = "catalog" if node.source == "catalog" else "cached implementation"
            print(
                f"  ◌  FGA  [{idx}/{n}] '{node.name}' — reusing from {where}",
                flush=True,
            )
            implemented.append(node)
            continue

        print(
            f"  ◌  FGA  [{idx}/{n}] implementing '{node.name}' with OpenAI...",
            flush=True,
        )

        test_feedback = ""
        code = ""
        deps: list[str] = []
        for attempt in range(1, 4):
            response = llm.invoke([
                SystemMessage(content=_OPENAI_FGA_SYSTEM),
                HumanMessage(content=_openai_turn_prompt(context_dir, node, test_feedback)),
            ])
            record_usage(response, "FGA")
            try:
                result = _extract_json_object(response.content)
            except Exception as exc:
                test_feedback = f"Your previous response was not valid JSON: {exc}"
                continue

            code = str(result.get("code") or "").strip()
            deps = clean_dependencies(result.get("dependencies") or [])
            if not code:
                test_feedback = "The JSON did not include non-empty Python source under 'code'."
                continue

            impl_path = context_dir / "functions" / f"{node.name}.py"
            impl_path.write_text(code)
            (context_dir / "functions" / f"{node.name}.deps.txt").write_text(
                "\n".join(deps) + ("\n" if deps else "")
            )

            ok, test_output = _run_stub_test(context_dir, node)
            if ok:
                break
            if _module_missing_due_to_reported_dep(test_output, deps):
                print(
                    f"       → local stub test needs generated dependency install: {', '.join(deps)}",
                    flush=True,
                )
                break
            if attempt == 3:
                raise RuntimeError(
                    f"OpenAI FGA could not produce a passing stub test for {node.name}.\n"
                    f"Last output:\n{test_output}"
                )
            test_feedback = test_output

        used = _extract_secrets(code)
        secrets = list(node.secrets) + [s for s in used if s not in node.secrets]
        done = node.model_copy(update={
            "code": code,
            "secrets": secrets,
            "dependencies": deps,
        })
        implemented.append(done)

        n_lines = len(code.splitlines())
        secrets_note = f" · secrets: {', '.join(secrets)}" if secrets else ""
        print(f"       → {n_lines} lines{secrets_note}", flush=True)

    return implemented


def _turn_prompt(spec: FunctionSpec, first: bool) -> str:
    """Build the small per-function turn prompt (shared rules live in the system prompt)."""
    fn = spec.name
    signature, test_call = signature_and_test_call(spec)

    if getattr(spec, "rank", 1) > 1:
        rank_rule = (
            f"- THIS FUNCTION IS RANKED: it runs as {spec.rank} parallel instances. "
            "Call `r = faasr_rank()` (it reports THIS instance's own index) and use "
            f"`r['rank']` (1..{spec.rank}) to process ONLY this instance's shard. The "
            "input/output names in the spec contain a {rank} placeholder — substitute "
            "r['rank'] (e.g. input1.format(rank=r['rank'])). Implement ONE function, "
            "never copies.\n"
        )
    else:
        rank_rule = (
            "- faasr_rank() reports THIS function's OWN rank and is only valid when this "
            "function is itself ranked — which it is NOT. Do NOT call faasr_rank(). If you "
            "feed a ranked successor, write exactly the literal shard count stated in the "
            "spec's \"Fan-out\" section (do not derive it from faasr_rank()).\n"
        )

    prefix = ""
    if first:
        prefix = (
            "First read CONTEXT.md — the shared workflow context (data-integrity rules, "
            "FaaSr API, filename-matching rules, stub-testing protocol). It applies to "
            "every function you implement this session.\n\n"
        )

    prompt = (
        prefix
        + f"Implement the FaaSr serverless function `{fn}`.\n"
        + f"Read specs/{fn}.md for its full specification.\n"
        + f"- Signature must be exactly: `{signature}`\n"
        + rank_rule
        + f"- Write the implementation to functions/{fn}.py.\n"
        + "- Validate it through the stubs per the testing rules, with:\n"
        + f"    {test_call}\n"
        + f"- Then write functions/{fn}.deps.txt (third-party PyPI packages, one per "
        + "line; empty file if none).\n"
    )

    if spec.source == "user_provided" and spec.user_model_mode == "verbatim":
        prompt += (
            f"\n[USER PROVIDED MODEL — VERBATIM] functions/original_{fn}.py is the user's "
            f"ORIGINAL Python script/function and its code must NOT change. Copy it UNCHANGED "
            f"into functions/{fn}.py, add a FaaSr-compliant entry function `{fn}` that produces "
            f"its results ONLY by CALLING the preserved model, prove the entry function invokes "
            f"the model, verify output parity, and write test_output/parity_report.txt. Full "
            f"required steps: specs/{fn}.md → '[User Provided Model] — Wrap Verbatim'.\n"
        )
    elif spec.source == "user_provided":
        prompt += (
            f"\n[USER PROVIDED MODEL] functions/original_{fn}.py is the user's ORIGINAL Python "
            f"script/function. Do NOT write {fn} from scratch — REFACTOR the original's logic "
            f"into functions/{fn}.py, apply exactly the changes the spec/feedback requires, "
            f"VERIFY PARITY against the original, and write test_output/parity_report.txt. Full "
            f"required steps: specs/{fn}.md → '[User Provided Model] — Refactor & Verify Parity'.\n"
        )
    return prompt


@dataclass
class _TurnUsage:
    """Per-turn usage/cost, shaped like a ResultMessage for record_sdk_usage / tracer.

    In a multi-turn ClaudeSDKClient session each turn's ResultMessage reports
    SESSION-CUMULATIVE total_cost_usd (and usage counters), so per-function
    accounting must diff consecutive results.
    """
    usage: dict = field(default_factory=dict)
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0


def _turn_delta(result: ResultMessage, prev: ResultMessage | None) -> _TurnUsage:
    """Diff two consecutive ResultMessages into this turn's own usage/cost.

    Counters are expected to be cumulative; if a delta comes out negative (i.e. the
    SDK reported per-turn values after all), fall back to the raw current value.
    """
    def _usage_of(msg) -> dict:
        u = getattr(msg, "usage", None) or {}
        return u if isinstance(u, dict) else {}

    cur_usage = _usage_of(result)
    prev_usage = _usage_of(prev) if prev is not None else {}

    delta_usage: dict[str, int] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        cur = int(cur_usage.get(key, 0) or 0)
        before = int(prev_usage.get(key, 0) or 0)
        delta_usage[key] = cur - before if cur >= before else cur

    cur_cost = float(getattr(result, "total_cost_usd", 0.0) or 0.0)
    prev_cost = float(getattr(prev, "total_cost_usd", 0.0) or 0.0) if prev else 0.0
    cost = cur_cost - prev_cost if cur_cost >= prev_cost else cur_cost

    return _TurnUsage(
        usage=delta_usage,
        total_cost_usd=cost,
        duration_ms=int(getattr(result, "duration_ms", 0) or 0),
        num_turns=int(getattr(result, "num_turns", 0) or 0),
    )


async def _implement_all(
    ordered_nodes: list[FunctionSpec],
    context_dir: Path,
    model: str,
    sdk_env: dict[str, str],
) -> list[FunctionSpec]:
    """
    Implement every non-reused node in ONE multi-turn Claude Agent SDK session.

    One ClaudeSDKClient session serves the whole workflow — the shared context
    (system-prompt rules, CONTEXT.md, stubs, reference files, earlier functions)
    is tokenized once and later turns hit the prompt cache, instead of the old
    one-fresh-session-per-function fan-out that re-read everything N times.

    Returns the nodes in BFS order with code/secrets/dependencies filled in.
    """
    implemented: list[FunctionSpec] = []
    n = len(ordered_nodes)

    def _reuse(idx: int, node: FunctionSpec) -> None:
        where = "catalog" if node.source == "catalog" else "cached implementation"
        print(
            f"  ◌  FGA  [{idx}/{n}] '{node.name}' — reusing from {where}",
            flush=True,
        )
        implemented.append(node)

    def _is_reused(node: FunctionSpec) -> bool:
        return node.source in ("catalog", "cached") and bool(node.code)

    if all(_is_reused(node) for node in ordered_nodes):
        for idx, node in enumerate(ordered_nodes, 1):
            _reuse(idx, node)
        return implemented

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
        cwd=context_dir,
        permission_mode="acceptEdits",
        model=model,
        env=sdk_env,
        system_prompt={"type": "preset", "preset": "claude_code", "append": _SHARED_RULES},
        # Opus 4.8/4.7 reject the legacy thinking.type.enabled (400); adaptive is the only
        # on-mode they accept (Sonnet 4.6 supports it too, so one value works for both).
        # Requires claude-agent-sdk >= 0.2.x, which emits `--thinking adaptive`; older 0.1.x
        # had no --thinking flag and mapped this to a nonzero --max-thinking-tokens (enabled),
        # which crashed Opus. effort defaults to "high".
        thinking={"type": "adaptive"},
    )

    debug = _debug_enabled()
    prev_result: ResultMessage | None = None
    first_turn = True

    async with ClaudeSDKClient(options=options) as client:
        for idx, node in enumerate(ordered_nodes, 1):
            if _is_reused(node):
                _reuse(idx, node)
                continue

            if node.source == "user_provided" and node.code:
                action = (
                    "wrapping user-provided model (verbatim)"
                    if node.user_model_mode == "verbatim"
                    else "refactoring user-provided code"
                )
                print(
                    f"  ◌  FGA  [{idx}/{n}] '{node.name}' — {action} + verifying parity...",
                    flush=True,
                )
            elif node.source == "adapt" and node.code:
                seed = "cached implementation" if node.origin_kind == "cache" else "catalog"
                print(
                    f"  ◌  FGA  [{idx}/{n}] '{node.name}' — adapting from {seed}...",
                    flush=True,
                )
            else:
                print(
                    f"  ◌  FGA  [{idx}/{n}] implementing '{node.name}'...",
                    flush=True,
                )

            tracer = None
            if debug:
                from faasr_agents.tui.agent_trace import AgentTracer
                tracer = AgentTracer(node.name, node.source)
                tracer.start()

            await client.query(_turn_prompt(node, first=first_turn))
            first_turn = False

            last_result: ResultMessage | None = None
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    last_result = msg
                if tracer is not None:
                    tracer.handle(msg)

            turn_usage = None
            if last_result is not None:
                turn_usage = _turn_delta(last_result, prev_result)
                prev_result = last_result
                # Exact provider-computed cost for this function's code generation.
                record_sdk_usage(turn_usage, "FGA", model)

            if tracer is not None:
                tracer.finish(turn_usage)
                trace_dir = context_dir / "_trace"
                trace_dir.mkdir(exist_ok=True)
                (trace_dir / f"{node.name}.md").write_text(tracer.transcript())

            impl_path = context_dir / "functions" / f"{node.name}.py"
            if not impl_path.exists():
                raise RuntimeError(
                    f"FGA agent did not produce functions/{node.name}.py "
                    f"for function '{node.name}'"
                )
            code = impl_path.read_text()

            # Finalize secrets from the actual code (union with FCA-declared), so the
            # emitted Secrets list and the user-facing list match what the code reads.
            used = _extract_secrets(code)
            secrets = list(node.secrets) + [s for s in used if s not in node.secrets]

            # Dependencies = what the agent reported in functions/<name>.deps.txt (it does
            # the import→PyPI mapping itself; we just sanity-filter stdlib/faasr).
            deps = _read_reported_deps(context_dir, node.name)

            done = node.model_copy(update={
                "code": code,
                "secrets": secrets, "dependencies": deps,
            })
            implemented.append(done)

            n_lines = len(code.splitlines())
            secrets_note = f" · secrets: {', '.join(secrets)}" if secrets else ""
            print(f"       → {n_lines} lines{secrets_note}", flush=True)

    return implemented


def _read_reported_deps(context_dir: Path, fn: str) -> list[str]:
    """Read the third-party PyPI deps the FGA agent declared in functions/<fn>.deps.txt.

    The agent reports its own dependencies (it knows the real import→package mapping);
    we just read + sanity-filter (drop stdlib/faasr) here.
    """
    path = context_dir / "functions" / f"{fn}.deps.txt"
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return clean_dependencies([ln for ln in lines if ln and not ln.startswith("#")])


def _extract_secrets(code: str) -> list[str]:
    """Pull secret names out of faasr_secret("NAME") / 'NAME' calls in the code.

    The generated code is the source of truth for which secrets the function
    actually reads, so the emitted Secrets list always matches reality. Names are
    used verbatim — no coercion — so the declared name and the runtime lookup agree.
    """
    found: list[str] = []
    for raw in re.findall(r"faasr_secret\(\s*['\"]([^'\"]+)['\"]", code):
        if raw not in found:
            found.append(raw)
    return found


def fga_node(state: AgentState) -> dict:
    """
    Function Generation Agent node — single-session Claude Agent SDK edition.

    Builds the complete context directory once (shared CONTEXT.md, per-node
    specs/<fn>.md, FaaSr stubs, seeded functions/ and test_data/), then runs ONE
    multi-turn Claude Agent SDK session that implements every non-reused node in
    BFS topological order — one turn per function. Each turn the agent:
      - Reads specs/<fn>.md (shared context was read on the first turn)
      - Writes functions/<fn>.py and validates it with the local FaaSr stubs
      - Iterates until the function runs cleanly against the test data

    The fga_node() signature and return shape are identical to the previous
    implementation so LangGraph routing is unaffected.
    """
    spec = state["workflow_spec"]
    code_feedback = (state.get("code_feedback") or "").strip()
    context_files = state.get("context_files") or []
    user_request = (state.get("user_request") or "").strip()
    model = get_default_model()

    ordered_nodes = bfs_topological_sort(spec.nodes, spec.edges)

    # In debug mode keep the context dir on disk for post-run inspection
    # (CONTEXT.md, specs/, functions/*.py, test_output/, _trace/); otherwise it is
    # auto-deleted when the with-block exits.
    temp_root = usable_temp_dir()
    if _debug_enabled():
        _kept = tempfile.mkdtemp(prefix=f"faasr_fga_{spec.name}_", dir=temp_root)
        print(f"  ◌  FGA  debug: context dir kept at {_kept}", flush=True)
        dir_ctx = contextlib.nullcontext(_kept)
    else:
        dir_ctx = tempfile.TemporaryDirectory(
            prefix=f"faasr_fga_{spec.name}_",
            dir=temp_root,
        )

    with dir_ctx as tmpdir:
        context_dir = Path(tmpdir)
        generate_nodes = [
            nd for nd in ordered_nodes
            if not (nd.source in ("catalog", "cached") and nd.code)
        ]
        setup_context_directory(
            context_dir, spec, context_files,
            generate_nodes=generate_nodes,
            feedback=code_feedback,
            user_request=user_request,
        )
        if using_openai():
            implemented_nodes = _implement_all_openai(ordered_nodes, context_dir, model)
        else:
            sdk_env = _build_sdk_env()
            implemented_nodes = asyncio.run(
                _implement_all(ordered_nodes, context_dir, model, sdk_env)
            )

    # Restore original spec node ordering before returning
    name_to_impl = {n.name: n for n in implemented_nodes}
    final_nodes = [name_to_impl.get(n.name, n) for n in spec.nodes]
    updated_spec = spec.model_copy(update={"nodes": final_nodes})

    n_impl = sum(1 for node in final_nodes if node.source not in ("catalog", "cached"))
    n_reused = len(final_nodes) - n_impl
    summary = f"FGA implemented {n_impl} function(s)"
    if n_reused:
        summary += f", reused {n_reused} (catalog/cached)"

    return {
        "workflow_spec": updated_spec,
        "code_feedback": None,
        "messages": [AIMessage(content=summary)],
    }
