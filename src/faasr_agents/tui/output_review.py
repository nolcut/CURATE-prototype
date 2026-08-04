"""
Gate 5 — Post-run output review + optional save for reuse.

run_output_review(payload) -> str
  payload keys: gate, workflow_json, artifacts, logs, code_by_node, workflow_name, message

  Returns one of:
    "accept"          — close review, no saving
    "save:functions"  — save functions to catalog only
    "save:workflow"   — save entire workflow only
    "save:both"       — save both
    "<feedback text>" — change request; loops back to WCA
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from faasr_agents.faasr.artifacts import (
    build_artifact_list as _build_artifact_list,
    expand_ranked_artifacts as _expand_ranked_artifacts,
    init_s3 as _init_s3,
)


def _download_artifact(artifacts: list[dict], downloaded: dict, download_dir: Path,
                       s3, idx: int) -> str:
    """Download artifact `idx` into download_dir, caching in `downloaded`."""
    matches = [a for a in artifacts if a["index"] == idx]
    if not matches:
        return f"No artifact at index {idx}"
    if not s3:
        return "S3 not available — check S3_AccessKey and S3_SecretKey env vars."
    a = matches[0]
    print(f"  Downloading {a['s3_key']} ...")
    download_dir.mkdir(parents=True, exist_ok=True)
    local_path = str(download_dir / a["filename"])
    try:
        s3.download_object(a["s3_key"], local_path)
    except Exception as e:
        return f"Download failed: {e}"
    downloaded[idx] = local_path
    return f"Saved to: {local_path}"


def _read_artifact_text(artifacts: list[dict], downloaded: dict, download_dir: Path,
                        s3, idx: int, max_chars: int = 4000) -> str:
    """Download (if needed) artifact `idx` and return its contents for the WDA."""
    if idx not in downloaded:
        result = _download_artifact(artifacts, downloaded, download_dir, s3, idx)
        if idx not in downloaded:
            return result
    path = Path(downloaded[idx])
    try:
        data = path.read_bytes()
    except OSError as e:
        return f"Could not read {path.name}: {e}"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"({path.name}: binary file, {len(data)} bytes)"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated; {len(data)} bytes total)"
    return text


def _list_s3_folder(s3, prefix: str, max_keys: int = 200) -> str:
    """List the object keys under an S3 prefix, one per line, for the WDA tools."""
    if not s3:
        return "S3 not available — check S3_AccessKey and S3_SecretKey env vars."
    try:
        keys = s3.list_objects(prefix.strip())
    except Exception as e:
        return f"List failed: {e}"
    if not keys:
        return f"No objects found under prefix '{prefix.strip()}'."
    lines = keys[:max_keys]
    if len(keys) > max_keys:
        lines.append(f"... and {len(keys) - max_keys} more")
    return "\n".join(lines)


def _try_open(path: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path])
        else:
            os.startfile(path)
    except Exception as e:
        print(f"  Could not open: {e}")


# Single source of truth for the REPL command menu — rendered both as the
# on-screen help (_print_summary) and as meta-context for the WDA review agent
# (_build_llm_context), so the assistant knows what the user can actually do
# (e.g. that `request`/`r` loops a change back to the workflow agents).
REPL_COMMANDS: list[tuple[str, str]] = [
    ("list", "list artifacts"),
    ("download <n>", "download artifact by index"),
    ("open <n>", "download + open artifact"),
    ("code <n|fn>", "show generated code for a function"),
    ("logs <n|fn>", "show execution logs for a function"),
    ("json", "view the emitted workflow.json"),
    ("export <dir>", "write the evaluation bundle (prompt, workflow, costs, revisions, logs) to <dir>"),
    ("accept", "accept results, no saving"),
    ("save [functions|workflow|both]", "save functions/workflow for reuse (menu if no arg)"),
    ("request <text>", "request changes — loops back to the workflow agents to regenerate or modify the workflow"),
    ("<question>", "ask the WDA about the run — it can download/open artifacts, view code/logs, or stage a revision for you"),
]


def _build_llm_context(payload: dict) -> str:
    lines = ["WORKFLOW RUN CONTEXT", "=" * 50]
    lines.append(f"Workflow: {payload.get('workflow_name', '?')}")
    if payload.get("success", True):
        lines.append("RUN STATUS: succeeded")
    else:
        failed = payload.get("failed_functions") or []
        lines.append(f"RUN STATUS: FAILED (failed functions: {', '.join(failed) or 'unknown'})")

    code_by_node: dict[str, str] = payload.get("code_by_node", {})
    if code_by_node:
        lines.append("\nGenerated function code:")
        for fn_name, code in code_by_node.items():
            lines.append(f"\n--- {fn_name} ---")
            lines.append(code or "(none)")

    logs: str = payload.get("logs", "")
    if logs:
        lines.append("\nExecution logs (last 8000 chars):")
        lines.append(logs[-8000:] if len(logs) > 8000 else logs)

    artifacts: list[dict] = payload.get("artifacts", [])
    lines.append("\nOutput artifacts on S3:")
    if artifacts:
        for a in artifacts:
            lines.append(f"  [{a['index']}] {a['s3_key']}  ({a['format']})")
    else:
        lines.append("  (none found)")

    lines.append('\nAVAILABLE REPL COMMANDS (the user types these at the "> " prompt):')
    for syntax, desc in REPL_COMMANDS:
        lines.append(f"  {syntax} — {desc}")

    return "\n".join(lines)


def _resolve_fn(arg: str, fn_names: list[str]) -> str | None:
    """Resolve a `code`/`logs` argument to a function name.

    Accepts an integer index into fn_names, an exact name, or a unique substring.
    Returns None if it can't be resolved unambiguously.
    """
    arg = arg.strip()
    if arg.isdigit():
        idx = int(arg)
        return fn_names[idx] if 0 <= idx < len(fn_names) else None
    if arg in fn_names:
        return arg
    partial = [n for n in fn_names if arg in n]
    return partial[0] if len(partial) == 1 else None


def _print_fn_roster(fn_names: list[str]) -> None:
    print("  Functions:")
    if fn_names:
        for i, name in enumerate(fn_names):
            print(f"    [{i}] {name}")
    else:
        print("    (none)")


def run_wda_summary(payload: dict, s3=None, artifacts: list[dict] | None = None,
                    downloaded: dict | None = None) -> str | None:
    """WDA post-run summary: inspect real artifacts via read_artifact, print a
    natural-language account of the run, and (at most) RECOMMEND a revision.

    Runs after every executed workflow — the Gate-5 REPL calls it on success
    (sharing its S3 client and download cache), the WDA node calls it on
    execution failure. Never raises; returns the summary text or None.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from faasr_agents.llm import get_llm
    from faasr_agents.pricing import record_usage
    from faasr_agents.prompts.wda_prompts import WDA_SUMMARY_SYSTEM

    try:
        workflow_json = payload.get("workflow_json", {})
        if artifacts is None:
            if s3 is None:
                s3 = _init_s3(workflow_json)
            artifacts = _expand_ranked_artifacts(payload.get("artifacts", []), s3)
        if downloaded is None:
            downloaded = {}
        wf_name = payload.get("workflow_name", "workflow")
        download_dir = Path(f"downloads/{wf_name}").resolve()

        @tool
        def read_artifact(index: int) -> str:
            """Download (if needed) an output artifact and return its contents (text truncated; binary reports size)."""
            return _read_artifact_text(artifacts, downloaded, download_dir, s3, index)

        @tool
        def list_s3_folder(prefix: str) -> str:
            """List the object keys under an S3 prefix (folder), one per line — use it to check whether expected inputs/outputs actually exist in the bucket."""
            return _list_s3_folder(s3, prefix)

        summary_tools = [read_artifact, list_s3_folder]
        tools_by_name = {t.name: t for t in summary_tools}

        llm = get_llm().bind_tools(summary_tools)
        messages = [
            SystemMessage(content=f"{WDA_SUMMARY_SYSTEM}\n\n{_build_llm_context(payload)}"),
            HumanMessage(content="Summarize this run now."),
        ]

        print()
        print("  " + "-" * 56)
        print("  WDA run summary")
        print("  " + "-" * 56)

        response = None
        for _ in range(6):
            response = llm.invoke(messages)
            record_usage(response, "WDA")
            messages.append(response)
            if not getattr(response, "tool_calls", None):
                break
            for tc in response.tool_calls:
                fn_tool = tools_by_name.get(tc["name"])
                if fn_tool is None:
                    result = f"Unknown tool: {tc['name']}"
                else:
                    try:
                        result = fn_tool.invoke(tc["args"])
                    except Exception as e:
                        result = f"Tool error: {type(e).__name__}: {e}"
                arg_str = ", ".join(f"{v}" for v in tc["args"].values())
                trace = str(result).splitlines()[0] if str(result).strip() else "(empty)"
                print(f"  [WDA] {tc['name']}({arg_str}) -> {trace[:100]}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        content = getattr(response, "content", "") if response is not None else ""
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        text = (content or "").strip()
        if text:
            print()
            for line in text.splitlines():
                print(f"  {line}")
            print()
            return text
        return None
    except Exception as e:
        print(f"  (WDA summary unavailable: {e})")
        return None


def _print_summary_header(payload: dict) -> None:
    """Banner — the context shown above the WDA summary."""
    wf = payload.get("workflow_name", "?")
    msg = payload.get("message", "")

    print()
    print("=" * 60)
    print(f"  GATE 5 — OUTPUT REVIEW: {wf}")
    print("=" * 60)
    if msg:
        print(f"  {msg}")
    print()


def _print_summary_footer(payload: dict) -> None:
    """Function roster, output artifacts, then the command menu — shown below
    the WDA summary, so artifacts sit after the code roster."""
    fn_names = list(payload.get("code_by_node", {}).keys())
    artifacts: list[dict] = payload.get("artifacts", [])

    print()
    _print_fn_roster(fn_names)
    print()
    print("  Output artifacts:")
    if artifacts:
        for a in artifacts:
            print(f"    [{a['index']}] {a['s3_key']}  ({a['format']})")
    else:
        print("    (none found in workflow JSON)")
    print()
    print("  Commands:")
    width = max(len(syntax) for syntax, _ in REPL_COMMANDS)
    for syntax, desc in REPL_COMMANDS:
        print(f"    {syntax.ljust(width)}  — {desc}")
    print()


def run_output_review(payload: dict, export_fn=None) -> str:
    """Interactive post-run review REPL. Returns a decision string.

    export_fn(dir) — provided by the CLI — writes the per-run evaluation bundle
    into the given directory (created if missing) and returns the bundle path.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from faasr_agents.llm import get_llm
    from faasr_agents.pricing import record_usage
    from faasr_agents.prompts.wda_prompts import WDA_REVIEW_SYSTEM

    workflow_json = payload.get("workflow_json", {})
    code_by_node: dict[str, str] = payload.get("code_by_node", {})
    fn_names = list(code_by_node.keys())
    logs: str = payload.get("logs", "")
    artifacts: list[dict] = payload.get("artifacts", [])

    s3 = _init_s3(workflow_json)
    # Resolve {rank} placeholders against the real S3 objects before display so
    # the banner, `list`, and `download`/`open` all reference concrete files.
    artifacts = _expand_ranked_artifacts(artifacts, s3)
    payload["artifacts"] = artifacts

    _print_summary_header(payload)

    wf_name = payload.get("workflow_name", "workflow")
    # Absolute so the "Saved to:" path the user sees is unambiguous and clickable.
    download_dir = Path(f"downloads/{wf_name}").resolve()
    downloaded: dict[int, str] = {}

    from faasr_agents.tui.line_input import read_line

    # WDA post-run summary: inspect the real outputs and report before the REPL
    # starts; shares the S3 client and download cache with the commands below.
    # Only on the first visit — a `save` loops back into this REPL (first_visit
    # False), and regenerating the summary each time would re-download artifacts
    # and re-spend WDA tokens for no new information.
    wda_summary = None
    if payload.get("first_visit", True):
        wda_summary = run_wda_summary(payload, s3=s3, artifacts=artifacts, downloaded=downloaded)

    # Function roster + command menu go below the summary so the natural-language
    # readout leads and the actionable options sit closest to the input prompt.
    _print_summary_footer(payload)

    # ── shared action helpers (used by both typed commands and WDA tools) ─────
    def _do_download(idx: int) -> str:
        return _download_artifact(artifacts, downloaded, download_dir, s3, idx)

    def _do_open(idx: int) -> str:
        if idx not in downloaded:
            result = _do_download(idx)
            if idx not in downloaded:
                return result
        _try_open(downloaded[idx])
        return f"Opened: {downloaded[idx]}"

    def _do_view_code(fn_query: str) -> str:
        fn = _resolve_fn(fn_query, fn_names)
        if fn is None:
            return (f"Could not resolve '{fn_query.strip()}'. "
                    f"Functions: {', '.join(fn_names) or '(none)'}")
        from faasr_agents.tui.run_viewer import run_output_viewer
        run_output_viewer({fn: code_by_node.get(fn) or ""}, "")
        return f"Code viewer for '{fn}' was shown; the user has closed it."

    def _do_view_logs(fn_query: str = "") -> str:
        from faasr_agents.tui.run_viewer import run_output_viewer
        if not fn_query.strip():
            run_output_viewer({}, logs)
            return "Log viewer (full run log) was shown; the user has closed it."
        fn = _resolve_fn(fn_query, fn_names)
        if fn is None:
            return (f"Could not resolve '{fn_query.strip()}'. "
                    f"Functions: {', '.join(fn_names) or '(none)'}")
        fn_logs = "\n".join(ln for ln in logs.splitlines() if fn in ln) or logs
        run_output_viewer({}, fn_logs)
        return f"Log viewer for '{fn}' was shown; the user has closed it."

    # ── WDA review agent: tools + chat state ──────────────────────────────────
    # pending_request carries a user-confirmed revision out of the agent loop;
    # once set, the REPL returns it (same path as the typed `request` command).
    pending_request: list[str] = []

    @tool
    def download_artifact(index: int) -> str:
        """Download an output artifact from S3 to the local downloads folder, by artifact index."""
        return _do_download(index)

    @tool
    def open_artifact(index: int) -> str:
        """Download (if needed) and open an output artifact with the system viewer, by artifact index."""
        return _do_open(index)

    @tool
    def read_artifact(index: int) -> str:
        """Download (if needed) an output artifact and return its contents (text truncated; binary reports size)."""
        return _read_artifact_text(artifacts, downloaded, download_dir, s3, index)

    @tool
    def view_code(function: str) -> str:
        """Open the full-screen code viewer on a generated function (by name or roster index)."""
        return _do_view_code(function)

    @tool
    def view_logs(function: str = "") -> str:
        """Open the full-screen log viewer; pass a function name to filter, or "" for the full run log."""
        return _do_view_logs(function)

    @tool
    def list_s3_folder(prefix: str) -> str:
        """List the object keys under an S3 prefix (folder), one per line — use it to check whether expected inputs/outputs actually exist in the bucket."""
        return _list_s3_folder(s3, prefix)

    @tool
    def revise_workflow(request: str) -> str:
        """Stage a workflow change request. The user is asked to confirm; if they do, the request
        is sent back to the workflow-composition agents to regenerate and redeploy the workflow."""
        request = request.strip()
        if not request:
            return "Empty request — nothing staged."
        print()
        print(f"  Revision request: {request}")
        try:
            confirm = read_line("  Send this back to the workflow agents? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = ""
        if confirm in ("y", "yes"):
            pending_request.append(request)
            return "User confirmed — the request will be sent to the workflow agents."
        return "User declined — the request was not sent."

    wda_tools = [download_artifact, open_artifact, read_artifact, list_s3_folder,
                 view_code, view_logs, revise_workflow]
    tools_by_name = {t.name: t for t in wda_tools}
    llm = get_llm().bind_tools(wda_tools)
    context_str = _build_llm_context(payload)
    if wda_summary:
        context_str += ("\n\nWDA RUN SUMMARY (you already presented this to the user):\n"
                        + wda_summary)
    messages = [SystemMessage(content=f"{WDA_REVIEW_SYSTEM}\n\n{context_str}")]
    MAX_TOOL_ROUNDS = 6

    while True:
        try:
            raw = read_line("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "accept"

        if not raw:
            continue

        cmd = raw.lower()
        parts = raw.split(maxsplit=1)
        cmd0 = parts[0].lower()

        # ── list ──────────────────────────────────────────────────
        if cmd == "list":
            if not artifacts:
                print("  (no artifacts)")
            else:
                for a in artifacts:
                    dl = f"  [local: {downloaded[a['index']]}]" if a["index"] in downloaded else ""
                    print(f"    [{a['index']}] {a['s3_key']}  ({a['format']}){dl}")

        # ── export ─────────────────────────────────────────────────
        elif cmd0 == "export":
            if export_fn is None:
                print("  Export is not available in this context.")
                continue
            if len(parts) < 2 or not parts[1].strip():
                print("  Usage: export <dir>")
                continue
            try:
                bundle = export_fn(parts[1].strip())
                print(f"  Evaluation bundle exported to: {bundle}")
            except Exception as e:
                print(f"  Export failed: {type(e).__name__}: {e}")

        # ── download ───────────────────────────────────────────────
        elif cmd0 == "download":
            try:
                idx = int(parts[1].strip()) if len(parts) > 1 else None
            except ValueError:
                idx = None
            if idx is None:
                print("  Usage: download <n>")
                continue
            print(f"  {_do_download(idx)}")

        # ── open ───────────────────────────────────────────────────
        elif cmd0 == "open":
            try:
                idx = int(parts[1].strip()) if len(parts) > 1 else None
            except ValueError:
                idx = None
            if idx is None:
                print("  Usage: open <n>")
                continue
            print(f"  {_do_open(idx)}")

        # ── code ───────────────────────────────────────────────────
        elif cmd0 == "code":
            if len(parts) < 2:
                print("  Usage: code <n|fn>")
                _print_fn_roster(fn_names)
                continue
            result = _do_view_code(parts[1])
            if result.startswith("Could not resolve"):
                print(f"  Could not resolve '{parts[1].strip()}'.")
                _print_fn_roster(fn_names)

        # ── logs ───────────────────────────────────────────────────
        elif cmd0 == "logs":
            result = _do_view_logs(parts[1] if len(parts) > 1 else "")
            if result.startswith("Could not resolve"):
                print(f"  Could not resolve '{parts[1].strip()}'.")
                _print_fn_roster(fn_names)

        # ── workflow.json ──────────────────────────────────────────
        elif cmd0 in ("json", "workflow"):
            import json as _json
            from faasr_agents.tui.run_viewer import run_output_viewer
            pretty = _json.dumps(workflow_json, indent=2) if workflow_json else ""
            if not pretty:
                print("  (no workflow.json available)")
            else:
                run_output_viewer({}, "", workflow_json=pretty)

        # ── accept ─────────────────────────────────────────────────
        elif cmd == "accept":
            print("  Accepted — no saving.")
            return "accept"

        # ── save ───────────────────────────────────────────────────
        elif cmd0 == "save":
            _SAVE = {
                "functions": ("save:functions", "Saving functions to catalog..."),
                "workflow": ("save:workflow", "Saving workflow to registry..."),
                "both": ("save:both", "Saving functions + workflow..."),
            }
            choice = parts[1].strip().lower() if len(parts) > 1 else ""
            if not choice:
                print()
                print("  What would you like to save?")
                print("    functions — individual functions to catalog")
                print("    workflow  — entire workflow for reuse")
                print("    both      — both")
                print("    cancel    — cancel")
                print()
                try:
                    choice = read_line("  Save choice: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
            if choice in _SAVE:
                decision, msg = _SAVE[choice]
                print(f"  {msg}")
                return decision
            print("  Save cancelled.")

        # ── request / changes ──────────────────────────────────────
        elif cmd0 in ("request", "changes"):
            if len(parts) < 2:
                print("  Usage: request <description of changes needed>")
                continue
            feedback = parts[1].strip()
            if not feedback:
                print("  Please describe the changes you want.")
                continue
            print(f"  Looping back to WCA: {feedback[:80]}...")
            return feedback

        # ── WDA review agent (tool-using Q&A) ──────────────────────
        else:
            try:
                messages.append(HumanMessage(content=raw))
                for _ in range(MAX_TOOL_ROUNDS):
                    response = llm.invoke(messages)
                    record_usage(response, "WDA")
                    messages.append(response)
                    if not getattr(response, "tool_calls", None):
                        break
                    for tc in response.tool_calls:
                        fn_tool = tools_by_name.get(tc["name"])
                        if fn_tool is None:
                            result = f"Unknown tool: {tc['name']}"
                        else:
                            try:
                                result = fn_tool.invoke(tc["args"])
                            except Exception as e:
                                result = f"Tool error: {type(e).__name__}: {e}"
                        arg_str = ", ".join(f"{v}" for v in tc["args"].values())
                        print(f"  [WDA] {tc['name']}({arg_str}) -> {result}")
                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                content = response.content
                if isinstance(content, list):
                    # Anthropic content blocks: keep only the text blocks
                    content = "\n".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                text = (content or "").strip()
                if text:
                    print()
                    for line in text.splitlines():
                        print(f"  {line}")
                    print()
                if pending_request:
                    print(f"  Looping back to WCA: {pending_request[0][:80]}...")
                    return pending_request[0]
            except Exception as e:
                print(f"  LLM error: {e}")
