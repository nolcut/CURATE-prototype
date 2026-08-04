"""
faasr-agents CLI — Human-in-the-loop driver for the FaaSr agent system.

Usage:
    faasr-agents                       # prompts interactively
    faasr-agents "describe workflow"   # description as argument
    echo "description" | faasr-agents # piped input

Flags:
    --opus / --sonnet   # model tier (default: sonnet)
    --anthropic-api     # use the Anthropic API (requires ANTHROPIC_API_KEY; default: Bedrock)
    --debug             # trace FGA agent tool calls + keep context dir on disk

The Gate-5 review REPL offers `export <dir>` to write a per-run evaluation
bundle (prompt, workflow, costs, revisions, adaptation decisions, logs).
"""
from __future__ import annotations
import os
import sys
import uuid

# Importing readline transparently upgrades the built-in input() to a full
# line editor: left/right arrows, mid-line editing, and up/down history — so
# prompts feel like a normal console. Guarded for platforms without it.
try:
    import readline  # noqa: F401
except ImportError:
    pass

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# ANSI colour helpers (degrade gracefully if terminal doesn't support them)
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)
CYAN   = lambda t: _c("96",   t)
GREEN  = lambda t: _c("92",   t)
YELLOW = lambda t: _c("93",   t)
RED    = lambda t: _c("91",   t)
BLUE   = lambda t: _c("94",   t)
MAGENTA= lambda t: _c("95",   t)

TERM_WIDTH = 72

GATE_META = {
    1: ("◆", CYAN,    "Workflow Skeleton",               "Review the proposed DAG. Approve or describe changes."),
    2: ("◆", CYAN,    "Function Candidates & Data Flow", "Review function sources and resolved file names."),
    3: ("◆", MAGENTA, "Code Review",                     "Browse the generated Python. Approve or describe changes."),
    4: ("◆", YELLOW,  "Deploy Approval",                 "Approve to upload code and trigger on GitHub Actions."),
    5: ("◆", GREEN,   "Output Review & Save",            "Inspect workflow outputs, ask questions; accept, save, or request changes."),
}

NODE_LABELS = {
    "compose_skeleton":    ("WCA",  "composing workflow skeleton from your request"),
    "gate1_skeleton":      ("WCA",  "waiting for Gate 1 approval"),
    "fca_resolve":         ("FCA",  "searching catalog and resolving data flow"),
    "gate2_candidates":    ("WCA",  "waiting for Gate 2 approval"),
    "fga_generate":        ("FGA",  "generating functions"),
    "gate3_code_review":   ("Code Review", "waiting for code approval"),
    "gate4_deploy":        ("WDA",  "validating workflow and preparing deployment"),
    "deploy_execute":      ("WDA",  "deploying and monitoring execution"),
    "gate5_output_review": ("Review", "inspecting workflow outputs"),
}

FAILURE_CLASS_LABELS = {
    "validation":     RED("Schema validation"),
    "infrastructure": YELLOW("Infrastructure / configuration"),
    "execution":      RED("Runtime execution"),
    "max_iterations": RED("Max revisions reached"),
    "user_declined":  DIM("User declined"),
    None:             GREEN("None"),
}


# ── layout helpers ────────────────────────────────────────────────────────────

def _banner() -> None:
    print()
    print(BOLD(CYAN("  ╔═══════════════════════════════╗")))
    print(BOLD(CYAN("  ║  FaaSr Agent System  v0.1     ║")))
    print(BOLD(CYAN("  ╚═══════════════════════════════╝")))
    print()


def _rule(label: str = "", char: str = "─") -> None:
    if label:
        side = (TERM_WIDTH - len(label) - 2) // 2
        print(DIM(char * side) + f" {label} " + DIM(char * (TERM_WIDTH - side - len(label) - 2)))
    else:
        print(DIM(char * TERM_WIDTH))


def _emit_cost_report(final_state: dict, thread_id: str) -> None:
    """Persist the per-agent cost report to JSON.

    No cost table is printed to the terminal — the breakdown lives only in the
    persisted costs_<thread_id>.json. FGA's code-gen cost is exact (Claude Agent
    SDK); every other agent is a token×price estimate. See faasr_agents.pricing.
    """
    import json
    import re
    from datetime import datetime
    from faasr_agents import pricing

    # Every real LLM call this run, including Gate-1/Gate-2 revision rounds that
    # LangGraph replays on interrupt() resume (not captured by graph state).
    records = pricing.get_run_records()
    if not records:
        return

    spec = final_state.get("workflow_spec")
    workflow_name = getattr(spec, "name", None) or "workflow"
    agents, total = pricing.summarize_run(records)

    # ── persist a structured record ──────────────────────────────────────────
    try:
        from faasr_agents.llm import get_default_model
        model_tier = get_default_model()
    except Exception:
        model_tier = ""

    payload = {
        "workflow": workflow_name,
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_tier,
        "total": total,
        "by_agent": agents,
        "records": [r.model_dump() for r in records],
    }
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", workflow_name).strip("_") or "workflow"
    out_dir = os.path.join(os.getcwd(), "faasr_output", safe_name)
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"costs_{thread_id}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    {DIM('Cost report saved to:')} {out_path}")
    except OSError as e:
        print(DIM(f"    (Could not write cost report: {e})"))


def _gate_header(gate: int) -> None:
    icon, colour, title, subtitle = GATE_META.get(gate, ("◆", CYAN, f"Gate {gate}", ""))
    print()
    _rule(char="━")
    print(f"  {colour(icon + '  Gate ' + str(gate))}  {BOLD(title)}")
    if subtitle:
        print(f"     {DIM(subtitle)}")
    _rule(char="━")
    print()


def _print_dag(spec_dict: dict | None, show_candidates: bool = False) -> None:
    """Render the workflow DAG with level-based layout, handling branching and fan-in."""
    if not spec_dict:
        return

    nodes = spec_dict.get("nodes", [])
    edges = spec_dict.get("edges", [])
    data_flow = spec_dict.get("data_flow", [])

    if not nodes:
        return

    name_to_node = {n["name"]: n for n in nodes}

    # Build adjacency
    successors: dict[str, list[str]] = {n["name"]: [] for n in nodes}
    predecessors: dict[str, list[str]] = {n["name"]: [] for n in nodes}
    for e in edges:
        fn, tn = e.get("from_node", ""), e.get("to_node", "")
        if fn not in successors or tn not in predecessors:
            continue  # dangling edge — render the rest rather than crash the gate
        if tn not in successors[fn]:
            successors[fn].append(tn)
        if fn not in predecessors[tn]:
            predecessors[tn].append(fn)

    # BFS level assignment (longest path to handle diamonds correctly)
    from collections import deque
    levels: dict[str, int] = {}
    sources = [n["name"] for n in nodes if not predecessors[n["name"]]]
    if not sources:
        sources = [nodes[0]["name"]]
    for s in sources:
        levels[s] = 0
    q: deque[str] = deque(sources)
    while q:
        name = q.popleft()
        for succ in successors[name]:
            new_lvl = levels[name] + 1
            if succ not in levels or levels[succ] < new_lvl:
                levels[succ] = new_lvl
                q.append(succ)
    # Fill any unvisited nodes
    for n in nodes:
        if n["name"] not in levels:
            levels[n["name"]] = 0

    max_level = max(levels.values())
    level_groups: dict[int, list[str]] = {i: [] for i in range(max_level + 1)}
    for name, lvl in levels.items():
        level_groups[lvl].append(name)

    # Data flow lookup
    df_by_edge: dict[tuple[str, str], str] = {}
    for df in data_flow:
        df_by_edge[(df.get("from_node", ""), df.get("to_node", ""))] = df.get("file", "")

    print(f"  {BOLD('Workflow DAG')}")
    print()

    def _node_box(name: str) -> None:
        import textwrap
        node = name_to_node.get(name, {})
        desc = node.get("description", "")
        source = node.get("source", "new")
        umm = node.get("user_model_mode")
        user_label = f"[User Provided Model — {umm}]" if umm else "[User Provided Model]"
        src_tag = (
            f"  {GREEN('[catalog]')}" if source == "catalog"
            else f"  {CYAN('[cached]')}" if source == "cached"
            else f"  {BLUE('[adapt]')}" if source == "adapt"
            else f"  {MAGENTA(user_label)}" if source == "user_provided"
            else (f"  {YELLOW('[new]')}" if show_candidates else "")
        )
        origin_kind = node.get("origin_kind")
        if origin_kind:
            kc = CYAN if origin_kind == "cache" else GREEN if origin_kind == "catalog" else BLUE
            origin_name = str(node.get("origin_name") or "")
            src_tag += f"  {kc('← ' + origin_kind)} {BOLD(origin_name)}"
        deps = node.get("dependencies", [])
        rank = node.get("rank", 1) or 1
        rank_tag = f"  {YELLOW('×' + str(rank) + ' parallel (rank)')}" if rank > 1 else ""

        print(f"  {CYAN('┌─')} {BOLD(name)}{rank_tag}{src_tag}")
        if rank > 1:
            print(f"  {CYAN('│')}  {DIM(f'runs as {rank} ranked instances; successor runs once after all finish')}")
        if desc:
            for line in textwrap.wrap(desc, width=TERM_WIDTH - 6):
                print(f"  {CYAN('│')}  {DIM(line)}")
        inputs  = node.get("inputs",  [])
        outputs = node.get("outputs", [])
        if inputs:
            in_names = ", ".join(x["name"] if isinstance(x, dict) else str(x) for x in inputs)
            print(f"  {CYAN('│')}  {DIM('in : ')}{in_names}")
        if outputs:
            out_names = ", ".join(x["name"] if isinstance(x, dict) else str(x) for x in outputs)
            print(f"  {CYAN('│')}  {DIM('out: ')}{out_names}")
        if deps:
            print(f"  {CYAN('│')}  {BLUE('deps:')} {GREEN(', '.join(deps))}")
        secrets = node.get("secrets", [])
        if secrets:
            print(f"  {CYAN('│')}  {YELLOW('secrets: ' + ', '.join(secrets))}")
        print(f"  {CYAN('└─')}")

    for lvl in range(max_level + 1):
        group = level_groups[lvl]
        parallel = len(group) > 1

        if parallel:
            print(f"  {DIM('┄ parallel ┄' + '─' * (TERM_WIDTH - 16))}")
        for name in group:
            _node_box(name)
        if parallel:
            print(f"  {DIM('─' * TERM_WIDTH)}")

        # Edges out of this level
        if lvl < max_level:
            out_edges: list[tuple[str, str, str]] = []
            for name in group:
                for succ in successors[name]:
                    file_label = df_by_edge.get((name, succ), "")
                    out_edges.append((name, succ, file_label))

            if not out_edges:
                print()
                continue

            # Branch / fan-in annotations. "branches" (not "fan-out"): this counts
            # distinct SUCCESSOR NODES, which is a different thing from a node's
            # rank — the parallelism the rest of the system calls fan-out. Labelling
            # two successors "fan-out ×2" reads as ranked parallelism that isn't there.
            n_from = len({e[0] for e in out_edges})
            n_to   = len({e[1] for e in out_edges})
            branch_tag  = f"  {YELLOW('[branches ×' + str(n_to) + ']')}"  if n_to  > 1 else ""
            fanin_tag   = f"  {MAGENTA('[fan-in ×'  + str(n_from) + ']')}" if n_from > 1 else ""

            print(f"  {CYAN('│')}")
            for frm, to, file_label in out_edges:
                edge_str = f"{frm} → {to}"
                if file_label:
                    print(f"  {CYAN('├──')} {DIM(file_label)}  {DIM(edge_str)}")
                else:
                    print(f"  {CYAN('├──')} {DIM(edge_str)}")
            print(f"  {CYAN('▼')}" + branch_tag + fanin_tag)
            print()

    print()


def _wrap(text: str, width: int = 62) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width)


def _print_deploy_summary(workflow_json: dict) -> None:
    name = workflow_json.get("WorkflowName", "?")
    entry = workflow_json.get("FunctionInvoke", "?")
    actions = workflow_json.get("ActionList", {})
    servers = workflow_json.get("ComputeServers", {})

    print(f"  {BOLD('Deployment summary')}")
    print()
    print(f"  {'Workflow:':<16} {name}")
    print(f"  {'Entry point:':<16} {entry}")
    print(f"  {'Functions:':<16} {len(actions)}")

    for sname, server in servers.items():
        platform = server.get("FaaSType", "?")
        repo     = server.get("ActionRepoName", "")
        print(f"  {'Platform:':<16} {platform}  {DIM(repo)}")

    print()
    print(f"  {BOLD('Action graph')}")
    for aname, action in actions.items():
        fn = action.get("FunctionName", aname)
        nxt = action.get("InvokeNext", [])
        arrow = f"  → {', '.join(str(n) for n in nxt)}" if nxt else "  (sink)"
        print(f"    {CYAN(aname)} {DIM(f'[{fn}]')}{DIM(arrow)}")
    print()

    _print_required_secrets(workflow_json)


def _print_external_inputs(items: list) -> None:
    """Visual list of files the user must place in S3 before running (external inputs)."""
    if not items:
        return
    print(f"  {BOLD(YELLOW('⬇  External inputs'))}  {DIM('— place these files in S3 before running')}")
    print()
    for it in items:
        path = it.get("s3_path", it.get("file", "?"))
        typ = it.get("type") or ""
        type_tag = f"  {DIM('(' + typ + ')')}" if typ else ""
        print(f"    {YELLOW('•')} {BOLD(path)}{type_tag}")
        desc = (it.get("description") or "").strip()
        for line in _wrap(desc, width=66):
            print(f"        {DIM(line)}")
        if it.get("node"):
            print(f"        {DIM('used by: ' + it['node'])}")
        print()


def _print_required_secrets(workflow_json: dict) -> None:
    """Visual list of GitHub repo secrets the user must add before the run."""
    secrets = workflow_json.get("Secrets", [])
    if not secrets:
        return
    print(f"  {BOLD(YELLOW('⚠  Secrets required'))}  {DIM('— add to your GitHub repo before deploying')}")
    print()
    for s in secrets:
        print(f"    {YELLOW('•')} {BOLD(s)}")
    print()
    print(f"    {DIM('Add via:')} Settings → Secrets and variables → Actions → New repository secret")
    print(f"    {DIM('Or via CLI:')}")
    for s in secrets:
        print(f"      {CYAN(f'gh secret set {s}')}")
    print()


# ── node progress ─────────────────────────────────────────────────────────────

def _node_progress(node_name: str, update: dict) -> None:
    if node_name.startswith("__"):
        return
    agent, action = NODE_LABELS.get(node_name, (node_name, "running"))
    # Extract a detail from messages if present
    detail = ""
    for m in reversed(update.get("messages", [])):
        content = getattr(m, "content", "") if hasattr(m, "content") else str(m)
        if content and not content.startswith("{"):
            detail = f"  {DIM('→ ' + content[:70])}"
            break
    print(f"  {DIM('◌')}  {BOLD(agent)}  {DIM(action)}{detail}")


# ── interrupt display ─────────────────────────────────────────────────────────

def _display_interrupt(value: dict, export_fn=None) -> str | None:
    """Display gate UI. Returns a decision string for Gate 5 (REPL), None for all other gates."""
    gate = value.get("gate", 0)
    _gate_header(gate)

    # Revision-loop context: explain WHY Gate 2 appeared again
    iteration = value.get("iteration", 0)
    prior_feedback = value.get("prior_feedback", "")
    if gate == 2 and iteration > 0:
        print(f"  {YELLOW('⚠')}  {BOLD(f'Revision loop — iteration {iteration}/5')}")
        print(f"  {DIM('WCA has revised the workflow per the pending request below.')}")
        print(f"  {DIM('Review the changes before approving.')}")
        print()
        if prior_feedback.strip():
            _rule("Pending revision request")
            for line in prior_feedback.strip().splitlines()[:20]:
                print(f"  {line}")
            print()

    msg = value.get("message", "")
    if msg and gate != 5:  # Gate 5 prints its own header via run_output_review
        for line in msg.strip().splitlines():
            print(f"  {line}")
        print()

    # Gate 1 & 2: show the DAG. At Gate 2 the DAG also carries the resolved
    # function-candidate details (source, secrets), so no separate summary.
    spec_dict = value.get("workflow_spec")
    if spec_dict:
        at_gate_2 = gate == 2          # Gate 2 shows resolved candidate detail in the DAG
        _print_dag(spec_dict, show_candidates=at_gate_2)

    # Gate 2 & 4: external inputs the user must provide in S3
    if gate in (2, 4):
        _print_external_inputs(value.get("external_inputs") or [])

    # Gate 3: interactive code-review browser, then the external-inputs reminder
    if gate == 3:
        review = value.get("review", [])
        if review:
            import json as _json
            from faasr_agents.tui.code_review import run_code_review
            wf_json = value.get("workflow_json") or {}
            run_code_review(
                review,
                workflow_json=_json.dumps(wf_json, indent=2) if wf_json else "",
            )
        _print_external_inputs(value.get("external_inputs") or [])

    # Gate 4: show deploy plan
    workflow_json = value.get("workflow_json")
    if workflow_json and gate == 4:
        _print_deploy_summary(workflow_json)

    # Gate 5: interactive output-review REPL — returns a decision string
    if gate == 5:
        from faasr_agents.tui.output_review import run_output_review
        return run_output_review(value, export_fn=export_fn)

    return None


def _prompt_user(gate: int) -> str:
    labels = {
        1: f"  {CYAN('▶')} Approve or describe changes",
        2: f"  {CYAN('▶')} Approve or describe changes",
        3: f"  {CYAN('▶')} Approve or describe changes",
        4: f"  {YELLOW('▶')} 'approve' / 'skip' / describe changes",
    }
    prompt = labels.get(gate, f"  {CYAN('▶')} Your response")
    print(prompt)
    response = input(f"  {BOLD('>')} ").strip()
    return response


# ── main ──────────────────────────────────────────────────────────────────────

_EDITOR_HINT = (
    "# Describe the scientific workflow you want to build.\n"
    "# Lines starting with '#' are ignored. Save and close the editor when done.\n"
    "\n"
)


def _read_description_via_editor():
    """Open $EDITOR (vim/vi/nano) for a multi-line workflow description.

    Returns the entered text (comment lines stripped), or None to signal the
    caller should fall back (no editor available or stdin is not a TTY).
    """
    import subprocess
    import tempfile
    import shutil as _shutil

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        editor = next((e for e in ("vim", "vi", "nano") if _shutil.which(e)), None)
    if not editor or not sys.stdin.isatty():
        return None

    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(_EDITOR_HINT)
        path = f.name
    cmd = [editor]
    if os.path.basename(editor) in ("vim", "vi", "nvim", "view"):
        # Don't let file content (hint or pasted text) be parsed as a vim modeline.
        cmd += ["--cmd", "set nomodeline"]
    cmd.append(path)
    try:
        subprocess.call(cmd)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    return body.strip()


def _sanitize_node_name(stem: str) -> str:
    """Turn a filename stem into a valid snake_case node identifier."""
    import re
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_").lower()
    if not s:
        return "user_fn"
    return ("fn_" + s) if s[0].isdigit() else s


def _attach_user_models() -> list:
    """Prompt for existing Python scripts/functions to attach to a new run.

    Each attached .py file becomes one [User Provided Model] node (name derived from
    the filename stem). Blank line finishes. Returns a list of UserModel (possibly
    empty). Bad paths are skipped with a warning — never fatal.
    """
    import ast
    from pathlib import Path
    from faasr_agents.models import UserModel

    print(f"  {BOLD('Attach existing Python scripts/functions?')} "
          f"{DIM('(enter a path, blank to finish)')}")
    models: list = []
    seen: set[str] = set()
    while True:
        raw = input(f"  {BOLD('+')} ").strip()
        if not raw:
            break
        p = Path(raw.strip("'\"")).expanduser()
        if not p.is_file() or p.suffix != ".py":
            print(f"  {YELLOW('skip')} {DIM(f'{p} — not a .py file')}")
            continue
        try:
            code = p.read_text()
        except OSError as ex:
            print(f"  {YELLOW('skip')} {DIM(f'{p} — {ex}')}")
            continue
        name = _sanitize_node_name(p.stem)
        if name in seen:
            print(f"  {YELLOW('skip')} {DIM(f'{p} — duplicate node name {name!r}')}")
            continue
        description = ""
        try:
            description = (ast.get_docstring(ast.parse(code)) or "").strip()
        except SyntaxError:
            pass  # non-fatal: a script may not even parse cleanly here — attach anyway
        models.append(UserModel(name=name, code=code, description=description, path=str(p)))
        seen.add(name)
        print(f"  {GREEN('✓')} {BOLD(name)} {DIM('[User Provided Model]')}")
    print()
    return models


def _attach_context_files() -> list:
    """Prompt for non-code reference files (papers, I/O examples, datasets, docs).

    Unlike the code prompt, anything entered here is treated as reference material,
    NOT a workflow node: each file is copied into the FGA's context/ folder and its
    name is surfaced to the planners. Any file type is accepted. Blank line finishes.
    Returns a list of ContextFile (possibly empty). Bad paths are skipped with a warning.
    """
    from pathlib import Path
    from faasr_agents.models import ContextFile

    print(f"  {BOLD('Attach data / reference files?')} "
          f"{DIM('(papers, I/O examples, datasets — enter a path, blank to finish)')}")
    files: list = []
    seen: set[str] = set()
    while True:
        raw = input(f"  {BOLD('+')} ").strip()
        if not raw:
            break
        p = Path(raw.strip("'\"")).expanduser()
        if not p.is_file():
            print(f"  {YELLOW('skip')} {DIM(f'{p} — not a file')}")
            continue
        if p.name in seen:
            print(f"  {YELLOW('skip')} {DIM(f'{p} — duplicate name {p.name!r}')}")
            continue
        files.append(ContextFile(name=p.name, path=str(p.resolve())))
        seen.add(p.name)
        print(f"  {GREEN('✓')} {BOLD(p.name)} {DIM('[User Provided Context]')}")
    print()
    return files


def _select_workflow_or_request():
    """Interactive startup: offer stored workflows for reuse, else compose a new one.

    Returns (reuse_spec | None, user_request, user_models, context_files).  Pick a
    stored workflow by number to reuse it (skips Gates 1 & 2, no attachments); press
    Enter to compose a new one in the editor, then optionally attach existing Python
    scripts/functions as [User Provided Model] nodes and any data/reference files.
    """
    from faasr_agents.catalog.workflow_store import WorkflowRegistry

    registry = WorkflowRegistry()
    stored = registry.list_all()

    if stored:
        print(f"  {BOLD('Stored workflows:')}")
        for i, e in enumerate(stored, 1):
            s = e.workflow_spec
            d = (s.description or "").strip().replace("\n", " ")
            print(f"    {CYAN(f'[{i}]')} {BOLD(s.name)} ({len(s.nodes)} fns)")
            for line in _wrap(d, width=64):
                print(f"        {DIM(line)}")
        print()
        print(f"  {BOLD('Pick a number to reuse, or press Enter to compose a new workflow:')}")
        print()
        raw = input(f"  {BOLD('>')} ").strip()
        print()
        if raw.isdigit() and 1 <= int(raw) <= len(stored):
            entry = stored[int(raw) - 1]
            registry.increment_usage(entry.id)
            spec = entry.workflow_spec
            # Reused functions already have code — they are this workflow's own
            # cached implementations, not freshly generated, so they should show
            # as [cached], not [new], at Gate 4.
            spec = spec.model_copy(update={"nodes": [
                node.model_copy(update={"source": "cached"}) if node.code else node
                for node in spec.nodes
            ]})
            # Reusing a stored workflow skips composition — no attachments.
            return spec, "", [], []
        # Anything other than a valid number → compose a new workflow in the editor.

    print(f"  {DIM('Opening an editor for the workflow description...')}")
    desc = _read_description_via_editor()
    if desc is None:
        # No editor / non-TTY: fall back to a single typed line.
        print(f"  {BOLD('Describe the scientific workflow you want to build:')}")
        print()
        desc = input(f"  {BOLD('>')} ").strip()
    print()

    user_models = _attach_user_models()
    context_files = _attach_context_files()
    return None, desc, user_models, context_files


def _is_transient_api_error(exc: Exception) -> bool:
    """Heuristically detect transient model-API errors (overload, throttle, 5xx)."""
    name = type(exc).__name__.lower()
    if any(k in name for k in (
        "overloaded", "apistatus", "ratelimit", "throttl",
        "serviceunavailable", "apiconnection", "apitimeout",
    )):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in (
        "overloaded", "529", "503", "throttl", "rate limit",
        "service unavailable", "too many requests",
    ))


def main() -> None:
    # Capture events + console from the very start (in memory, unconditional)
    # so the Gate-5 REPL's `export <dir>` command has the whole run narrative.
    from faasr_agents import export as run_export
    run_export.reset_events()
    run_export.install_console_capture()

    _banner()

    # Model selection flags: --opus (Opus 4.8) or --sonnet (Sonnet 4.6).
    # Parse and strip them before treating the rest of argv as the request.
    from faasr_agents.llm import set_model_tier, set_provider

    args = sys.argv[1:]

    tier = "sonnet"
    if "--opus" in args:
        tier = "opus"
    elif "--sonnet" in args:
        tier = "sonnet"

    # --anthropic-api opts into the direct Anthropic API. Without the flag the
    # run is Bedrock-only and ANTHROPIC_API_KEY is never used, even if set.
    anthropic_api = "--anthropic-api" in args
    if anthropic_api and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            RED("  Error: --anthropic-api requires ANTHROPIC_API_KEY (set it in .env)."),
            file=sys.stderr,
        )
        sys.exit(1)

    # --debug enables FGA agent tracing (tool calls + order) and keeps the
    # per-function context dir on disk for inspection. Bridged to fga.py via env.
    debug = "--debug" in args
    if debug:
        os.environ["FAASR_DEBUG"] = "1"

    args = [a for a in args if a not in ("--opus", "--sonnet", "--anthropic-api", "--debug")]
    set_model_tier(tier)
    if anthropic_api:
        set_provider("anthropic")
    print(f"  {DIM('Model tier: ' + tier)}")
    if anthropic_api:
        print(f"  {DIM('Provider: Anthropic API')}")
    if debug:
        print(f"  {DIM('Debug tracing: on')}")

    reuse_spec = None
    user_models = []  # attachments only come from the interactive picker
    context_files = []  # non-code reference files, likewise picker-only
    if args:
        user_request = " ".join(args)
    elif not sys.stdin.isatty():
        user_request = sys.stdin.read().strip()
    else:
        reuse_spec, user_request, user_models, context_files = _select_workflow_or_request()

    if reuse_spec is None and not user_request:
        print(RED("  Error: no workflow description provided."), file=sys.stderr)
        sys.exit(1)

    from faasr_agents.orchestrator import compile_graph

    _rule()
    if reuse_spec is not None:
        print(f"  {BOLD('Reusing stored workflow:')} {reuse_spec.name}")
        print(f"  {DIM('Skipping composition — going straight to deployment (Gate 4)...')}")
        app = compile_graph(entry_node="gate4_deploy")
        user_request = ""
    else:
        print(f"  {BOLD('Request:')} {user_request}")
        print(f"  {DIM('Thinking...')}")
        app = compile_graph()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_request": user_request,
        "workflow_spec": reuse_spec,
        "messages": [],
        "hitl_decision": None,
        "deploy_result": None,
        "feedback": None,
        "revision_directives": [],
        "iteration": 0,
        "hitl_gate": None,
        "code_feedback": None,
        "structural_change": None,
        "wca_fca_summary": None,
        "wca_enriched_spec": None,
        "review_continue": None,
        "prior_spec": None,
        "spec_is_seed": None,
        "user_models": user_models,
        "context_files": context_files,
    }

    # Start this run's cost accounting from empty (see faasr_agents.pricing).
    from faasr_agents import pricing
    pricing.reset_run_records()

    def export_fn(dir_arg: str):
        """Gate-5 REPL `export <dir>`: write the evaluation bundle from live state."""
        from faasr_agents.llm import selected_provider
        run_dir = run_export.make_run_dir(dir_arg)
        state_values = app.get_state(config).values
        return run_export.export_run(
            run_dir, state_values, thread_id,
            records=pricing.get_run_records(),
            events=run_export.get_events(),
            meta={"model_tier": tier, "provider": selected_provider, "debug": debug},
        )

    current_input = initial_state
    while True:
        print()
        try:
            for chunk in app.stream(current_input, config=config, stream_mode="updates"):
                for node_name, update in chunk.items():
                    _node_progress(node_name, update)
        except Exception as e:
            if _is_transient_api_error(e):
                print()
                print(f"  {RED('✘')}  {BOLD('The model API is temporarily overloaded.')}")
                print(f"  {DIM('Transient server-side error (already retried with backoff). Re-run in a moment.')}")
                print(f"  {DIM(f'Details: {type(e).__name__}: {str(e)[:140]}')}")
                sys.exit(1)
            raise

        state = app.get_state(config)

        # Look for a pending gate FIRST. A node that loops on interrupt()
        # (e.g. Gates 1 & 2 re-prompting after a "describe changes" response)
        # re-fires its interrupt with state.next == () — so checking state.next
        # before the interrupt would wrongly end the run on every refinement.
        interrupt_data = None
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_data = task.interrupts[0].value
                break

        # No gate waiting → the graph has finished (or stopped with nothing
        # actionable). Either way, fall through to the final result.
        if not (interrupt_data and isinstance(interrupt_data, dict)):
            break

        print()
        gate = interrupt_data.get("gate", 0)
        run_export.record_event("gate_shown", gate=gate,
                                iteration=interrupt_data.get("iteration"))
        # Gate 2's payload carries the FCA sourcing decisions for this
        # resolve round — capture it here (state nulls it on approve).
        if gate == 2 and interrupt_data.get("fca_summary") is not None:
            run_export.record_event("fca_decisions",
                                    summary=interrupt_data["fca_summary"])
        gate_decision = _display_interrupt(interrupt_data, export_fn=export_fn)

        # Gate 5: REPL returned a decision directly — no additional prompt needed
        if gate == 5 and gate_decision is not None:
            if gate_decision == "accept":
                print(f"  {DIM('Accepted — continuing...')}")
            elif gate_decision.startswith("save:"):
                print(f"  {DIM('Saved — continuing...')}")
            else:
                print(f"  {DIM('Looping back to WCA with your change request...')}")
            approve5 = gate_decision == "accept" or gate_decision.startswith("save:")
            run_export.record_event(
                "gate_decision", gate=5,
                action="approve" if approve5 else "revise",
                directive=None if approve5 else gate_decision,
            )
            print()
            current_input = Command(resume=gate_decision)
            continue

        user_response = _prompt_user(gate)
        print()
        approved = user_response.lower() in ("approve", "yes", "skip")
        if not approved:
            if gate == 3:
                print(f"  {DIM('Regenerating functions with your feedback...')}")
            else:
                print(f"  {DIM('Revising workflow...')}")
        else:
            print(f"  {DIM('Continuing...')}")
        run_export.record_event(
            "gate_decision", gate=gate,
            action="approve" if approved else "revise",
            directive=None if approved else user_response,
        )
        current_input = Command(resume=user_response)

    # ── final result ──────────────────────────────────────────────────────────
    final_state = app.get_state(config).values
    print()
    _rule("Result", char="═")
    print()

    deploy_result = final_state.get("deploy_result") or {}
    failure_class = deploy_result.get("failure_class")
    user_message  = deploy_result.get("user_message", "")
    artifacts_dir = deploy_result.get("artifacts_dir", "")
    success       = deploy_result.get("success", False)

    if success and deploy_result.get("logs") != "(skipped)":
        print(f"  {GREEN('✔')}  {BOLD('Workflow deployed and executed successfully.')}")
    elif success:
        # skipped or user_declined with success=True
        print(f"  {CYAN('○')}  {BOLD('Workflow composed and validated.')}")
    else:
        fc_label = FAILURE_CLASS_LABELS.get(failure_class, RED(str(failure_class)))
        print(f"  {RED('✘')}  {BOLD('Workflow did not complete.')}")
        print(f"  {DIM('Failure class:')} {fc_label}")

    if user_message:
        print()
        for line in user_message.strip().splitlines():
            prefix = f"  {YELLOW('│')}  " if not success else f"  {GREEN('│')}  "
            print(f"{prefix}{line}")

    if artifacts_dir:
        print()
        print(f"  {CYAN('▸')} Artifacts saved to:")
        print(f"       {BOLD(artifacts_dir)}")
        print(f"     {DIM('workflow.json  functions/  tests/')}")

    logs = deploy_result.get("logs", "")
    if logs and logs not in ("(skipped)", ""):
        print()
        _rule("Execution logs")
        for line in logs[:2000].splitlines():
            print(f"  {line}")

    spec = final_state.get("workflow_spec")
    if spec:
        print()
        _rule("Workflow summary")
        print()
        n_impl = sum(1 for n in spec.nodes if n.code)
        print(f"  {BOLD(spec.name)}")
        print(f"  {DIM(f'{len(spec.nodes)} functions  •  {n_impl} implemented')}")
        print()
        for node in spec.nodes:
            icon  = GREEN("✔") if node.code else DIM("○")
            umm = getattr(node, "user_model_mode", None)
            user_label = f"[User Provided Model — {umm}]" if umm else "[User Provided Model]"
            src   = (
                GREEN("[catalog]") if node.source == "catalog"
                else YELLOW("[adapt]") if node.source == "adapt"
                else CYAN("[cached]") if node.source == "cached"
                else MAGENTA(user_label) if node.source == "user_provided"
                else YELLOW("[new]")
            )
            ok = getattr(node, "origin_kind", None)
            origin = ""
            if ok:
                kc = CYAN if ok == "cache" else GREEN if ok == "catalog" else BLUE
                origin = f"{kc('← ' + ok)} {BOLD(node.origin_name or '')}"
            print(f"    {icon}  {BOLD(node.name)}  {src}{('  ' + origin) if origin else ''}")
            if node.description:
                import textwrap
                for line in textwrap.wrap(node.description, width=TERM_WIDTH - 10):
                    print(f"         {DIM(line)}")
        print()

    iteration = final_state.get("iteration", 0)
    if iteration > 0:
        print(f"  {DIM(f'Total revision cycles: {iteration}')}")
        print()

    _emit_cost_report(final_state, thread_id)
    print()

    _rule(char="═")
    print()


if __name__ == "__main__":
    main()
