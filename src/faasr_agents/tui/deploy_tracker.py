"""
Full-phase deployment TUI for faasr-agents.

Covers: step tracker (upload → register/invoke) → live execution monitor → executive summary.
Replaces the silent _monitor_execution loop in wda.py.

Usage::

    deploy = run_deploy_tui(spec, workflow_json, steps, build_runner)
    # deploy["stage"] in ("infra", "monitor")
    # deploy["success"] bool
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from faasr_agents.tui.styles import STATUS_STYLE


# ── Action view ───────────────────────────────────────────────────────────────

class ActionView:
    """Tracks which action is displayed; auto-follows unless the user cycled."""

    def __init__(self, action_names: list[str]):
        self._names = action_names
        self._idx = 0
        self._auto = True
        self._lock = threading.Lock()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._names[self._idx] if self._names else None

    @property
    def index(self) -> int:
        with self._lock:
            return self._idx

    def next(self) -> None:
        with self._lock:
            if self._names:
                self._idx = (self._idx + 1) % len(self._names)
                self._auto = False

    def prev(self) -> None:
        with self._lock:
            if self._names:
                self._idx = (self._idx - 1) % len(self._names)
                self._auto = False

    def auto_select(self, statuses: dict) -> None:
        """Follow the most active function unless the user has manually navigated."""
        from faasr_agents.faasr.runtime.enums import FunctionStatus
        with self._lock:
            if not self._auto or not self._names:
                return
            for i, name in enumerate(self._names):
                if statuses.get(name) == FunctionStatus.RUNNING:
                    self._idx = i
                    return
            for i in range(len(self._names) - 1, -1, -1):
                if statuses.get(self._names[i]) == FunctionStatus.COMPLETED:
                    self._idx = i
                    return


# ── Scroll state ──────────────────────────────────────────────────────────────

class ScrollState:
    """Per-action scroll offsets for both code and log panels."""

    def __init__(self):
        self._code: dict[str, int] = {}
        self._log:  dict[str, int] = {}
        self._lock = threading.Lock()
        # Updated each render cycle from console.height
        self.code_page = 20
        self.log_page  = 15

    def get_code(self, fn_name: str) -> int:
        with self._lock:
            return self._code.get(fn_name, 0)

    def get_log(self, fn_name: str) -> int:
        with self._lock:
            return self._log.get(fn_name, 0)

    def scroll_code_up(self, fn_name: str, amount: int = 5) -> None:
        with self._lock:
            self._code[fn_name] = max(0, self._code.get(fn_name, 0) - amount)

    def scroll_code_down(self, fn_name: str, total_lines: int, amount: int = 5) -> None:
        with self._lock:
            max_off = max(0, total_lines - self.code_page)
            self._code[fn_name] = min(max_off, self._code.get(fn_name, 0) + amount)

    def scroll_log_up(self, fn_name: str, amount: int = 5) -> None:
        with self._lock:
            self._log[fn_name] = max(0, self._log.get(fn_name, 0) - amount)

    def scroll_log_down(self, fn_name: str, amount: int = 5) -> None:
        """Increment freely; render-time clamp keeps it in bounds."""
        with self._lock:
            self._log[fn_name] = self._log.get(fn_name, 0) + amount


# ── Keyboard listener (monitor phase) ─────────────────────────────────────────

def _monitor_keyboard_listener(
    view: ActionView,
    scroll: ScrollState,
    stop: threading.Event,
    code_by_action: dict[str, str],
) -> None:
    """
    ← → / n p  — cycle actions
    ↑ ↓        — scroll code
    w / s      — scroll logs
    """
    try:
        import readchar
        from faasr_agents.faasr.runtime.utils import extract_function_name
    except ImportError:
        return
    try:
        while not stop.is_set():
            key = readchar.readkey()
            fn = view.current
            if key in (readchar.key.RIGHT, "n"):
                view.next()
            elif key in (readchar.key.LEFT, "p"):
                view.prev()
            elif key == readchar.key.UP:
                if fn:
                    scroll.scroll_code_up(fn)
            elif key == readchar.key.DOWN:
                if fn:
                    base = extract_function_name(fn)
                    total = len((code_by_action.get(base) or "").splitlines())
                    scroll.scroll_code_down(fn, total)
            elif key == "w":
                if fn:
                    scroll.scroll_log_up(fn)
            elif key == "s":
                if fn:
                    scroll.scroll_log_down(fn)
    except Exception:
        pass


# ── Step-phase rendering ──────────────────────────────────────────────────────

def _build_step_table(steps: list, step_states: list[str], workflow_name: str) -> Table:
    table = Table(
        title=f"[bold]{workflow_name}[/bold] — Deploying",
        expand=True,
        show_lines=True,
    )
    table.add_column("Step", ratio=4)
    table.add_column("Status", justify="center", ratio=1)

    _STATE = {
        "pending": ("dim",        "○  pending"),
        "running": ("yellow",     "⟳  running"),
        "done":    ("green bold", "✔  done"),
        "failed":  ("red bold",   "✘  failed"),
    }
    for (label, _), state in zip(steps, step_states):
        style, text = _STATE.get(state, ("", state))
        table.add_row(label, Text(text, style=style))

    return table


# ── Monitor-phase rendering ───────────────────────────────────────────────────

def _build_status_table(statuses: dict, workflow_name: str, viewed_idx: int) -> Table:
    table = Table(
        title=f"[bold]{workflow_name}[/bold]",
        expand=True,
        show_lines=True,
    )
    table.add_column("", width=1)
    table.add_column("Action", style="cyan", ratio=2)
    table.add_column("Status", justify="center", ratio=1)

    for i, (name, status) in enumerate(statuses.items()):
        style, label = STATUS_STYLE.get(status, ("", str(status.value)))
        indicator = "▶" if i == viewed_idx else " "
        table.add_row(indicator, name, Text(label, style=style))

    return table


def _progress_text(statuses: dict) -> str:
    from faasr_agents.faasr.runtime.enums import FunctionStatus
    done = sum(1 for s in statuses.values() if s == FunctionStatus.COMPLETED)
    return f"Progress: {done}/{len(statuses)} complete"


def _build_detail_panel(
    fn_name: str,
    runner: Any,
    code_by_action: dict[str, str],
    scroll: Optional[ScrollState] = None,
) -> Any:
    """Build scrollable log + code panels for the viewed action."""
    from faasr_agents.faasr.runtime.utils import extract_function_name

    log_page  = scroll.log_page  if scroll else 15
    code_page = scroll.code_page if scroll else 20

    # ── Logs ─────────────────────────────────────────────────────────────────
    try:
        logs = runner.get_function_logs_content(fn_name)
        log_lines = logs.splitlines() if logs else []
    except Exception:
        log_lines = ["(unable to read logs)"]

    log_total = len(log_lines)
    if scroll:
        log_offset = min(scroll.get_log(fn_name), max(0, log_total - log_page))
    else:
        log_offset = max(0, log_total - log_page)  # default: tail

    log_visible = log_lines[log_offset: log_offset + log_page] if log_lines else ["(no logs yet)"]
    log_top = log_offset + 1
    log_bot = min(log_offset + log_page, max(1, log_total))
    log_hint = f"  lines {log_top}–{log_bot}/{log_total}  w/s scroll" if log_total > log_page else ""
    log_panel = Panel(
        "\n".join(log_visible),
        title=f"Logs: {fn_name}{log_hint}",
        border_style="blue",
        expand=True,
    )

    # ── Code (always available locally) ─────────────────────────────────────
    base = extract_function_name(fn_name)
    code = code_by_action.get(base, "")
    if not code:
        return log_panel

    code_lines = code.splitlines()
    code_total = len(code_lines)
    code_offset = scroll.get_code(fn_name) if scroll else 0
    code_visible = code_lines[code_offset: code_offset + code_page]
    c_top = code_offset + 1
    c_bot = min(code_offset + code_page, code_total)
    code_hint = (
        f"  lines {c_top}–{c_bot}/{code_total}  ↑↓ scroll"
        if code_total > code_page
        else f"  {code_total} lines"
    )

    try:
        code_renderable = Syntax(
            "\n".join(code_visible),
            "python",
            theme="monokai",
            line_numbers=True,
            start_line=code_offset + 1,
        )
    except Exception:
        code_renderable = Text("\n".join(code_visible))

    code_panel = Panel(
        code_renderable,
        title=f"Code: {fn_name}{code_hint}",
        border_style="green",
        expand=True,
    )

    return Group(log_panel, code_panel)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_deploy_tui(
    spec: Any,
    workflow_json: dict,
    steps: list[tuple[str, Callable]],
    build_runner: Callable,
) -> dict:
    """
    Run the full deployment TUI and return a result dict.

    Args:
        spec: WorkflowSpec — used to build the action-name → local-code map.
        workflow_json: Emitted FaaSr JSON — provides WorkflowName.
        steps: Ordered list of (label, callable).  Each callable returns
               ``{"success": bool, "error": str|None, ...}``.  The step that
               invokes the workflow must also include ``"faasr_payload"``.
        build_runner: Callable(faasr_payload) -> started WorkflowRunner.

    Returns:
        On infrastructure failure::

            {"stage": "infra", "success": False, "error": str, "step_label": str|None}

        On monitor completion::

            {"stage": "monitor", "success": bool, "logs": str,
             "errors": str, "failed_functions": list, "invocation_id": str}
    """
    from faasr_agents.faasr.emit import _to_action_name
    from faasr_agents.faasr.runtime.enums import FunctionStatus

    workflow_name = workflow_json.get("WorkflowName", getattr(spec, "name", "Workflow"))

    code_by_action = {
        _to_action_name(n.name): n.code
        for n in spec.nodes
        if n.code
    }

    console = Console()

    # ── Phase 1: step tracker ─────────────────────────────────────────────────
    step_states = ["pending"] * len(steps)
    _lock = threading.Lock()
    payload_holder: dict = {"faasr_payload": None}
    infra_error: dict = {"error": None, "label": None}
    steps_done = threading.Event()

    def _run_steps() -> None:
        for i, (label, fn) in enumerate(steps):
            with _lock:
                step_states[i] = "running"
            try:
                result = fn()
            except Exception as exc:
                result = {"success": False, "error": str(exc)}

            with _lock:
                if result.get("success"):
                    step_states[i] = "done"
                    if result.get("faasr_payload"):
                        payload_holder["faasr_payload"] = result["faasr_payload"]
                else:
                    step_states[i] = "failed"
                    infra_error["error"] = result.get("error", "Unknown error")
                    infra_error["label"] = label
                    steps_done.set()
                    return
        steps_done.set()

    step_thread = threading.Thread(target=_run_steps, daemon=True)
    step_thread.start()

    with Live(console=console, refresh_per_second=4) as live:
        while not steps_done.is_set():
            with _lock:
                snapshot = list(step_states)
            live.update(Panel(
                _build_step_table(steps, snapshot, workflow_name),
                border_style="blue",
            ))
            time.sleep(0.25)
        with _lock:
            snapshot = list(step_states)
        live.update(Panel(
            _build_step_table(steps, snapshot, workflow_name),
            border_style="blue",
        ))

    if infra_error["error"]:
        console.print()
        return {
            "stage": "infra",
            "success": False,
            "error": infra_error["error"],
            "step_label": infra_error["label"],
        }

    faasr_payload = payload_holder["faasr_payload"]
    if not faasr_payload:
        return {
            "stage": "infra",
            "success": False,
            "error": "No workflow payload returned from deployment steps",
            "step_label": None,
        }

    # ── Phase 2: live execution monitor ──────────────────────────────────────
    # Suppress WorkflowRunner's StreamHandler while Live is active.
    wf_logger = logging.getLogger("WorkflowRunner")
    orig_level = wf_logger.level
    wf_logger.setLevel(logging.CRITICAL)

    try:
        runner = build_runner(faasr_payload)
    except Exception as exc:
        wf_logger.setLevel(orig_level)
        return {
            "stage": "infra",
            "success": False,
            "error": f"WorkflowRunner initialisation failed: {exc}",
            "step_label": "Initialise monitor",
        }

    invocation_id = getattr(runner, "invocation_id", "N/A")
    initial_statuses = runner.get_function_statuses()
    action_names = list(initial_statuses.keys())

    view   = ActionView(action_names)
    scroll = ScrollState()
    stop_event = threading.Event()

    kb_thread = threading.Thread(
        target=_monitor_keyboard_listener,
        args=(view, scroll, stop_event, code_by_action),
        daemon=True,
    )
    kb_thread.start()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="status", ratio=1),
        Layout(name="detail", ratio=2),
    )

    start_time = time.time()

    try:
        with Live(layout, console=console, refresh_per_second=2, screen=True):
            while not runner.monitoring_complete:
                statuses = runner.get_function_statuses()
                elapsed  = time.time() - start_time

                # Dynamic page sizes based on terminal height.
                # body ≈ console.height - header(3) - footer(3) - panel borders(4)
                body_h = max(20, console.height - 10)
                scroll.log_page  = max(6, body_h // 2 - 3)
                scroll.code_page = max(6, body_h // 2 - 3)

                view.auto_select(statuses)
                viewed_fn  = view.current
                viewed_idx = view.index

                layout["header"].update(Panel(
                    f"[bold]{workflow_name}[/bold]  |  "
                    f"Invocation: {invocation_id}  |  "
                    f"{_progress_text(statuses)}",
                    style="bold",
                ))
                layout["status"].update(
                    _build_status_table(statuses, workflow_name, viewed_idx)
                )
                if viewed_fn:
                    layout["detail"].update(
                        _build_detail_panel(viewed_fn, runner, code_by_action, scroll)
                    )
                else:
                    layout["detail"].update(
                        Panel("Waiting for functions to start...", title="Detail", border_style="dim")
                    )
                layout["footer"].update(Panel(
                    f"Elapsed: {elapsed:.0f}s  |  "
                    "[dim]← →  cycle actions  ·  ↑ ↓  scroll code  ·  w / s  scroll logs  ·  Ctrl+C  stop[/dim]",
                    style="dim",
                ))

                time.sleep(0.5)
    except KeyboardInterrupt:
        runner.shutdown()
    finally:
        stop_event.set()
        wf_logger.setLevel(orig_level)

    # ── Executive summary ─────────────────────────────────────────────────────
    statuses = runner.get_function_statuses()
    failed_functions = [
        name for name, st in statuses.items()
        if st in (FunctionStatus.FAILED, FunctionStatus.TIMEOUT)
    ]
    all_ok = not failed_functions

    from faasr_agents.faasr.runtime.utils import extract_function_name

    console.print()
    console.rule("[bold]Executive Summary[/bold]")
    console.print()
    console.print(_build_status_table(statuses, workflow_name, viewed_idx=-1))
    console.print()

    if all_ok:
        console.print("[green bold]✔  Workflow completed successfully.[/green bold]")
    else:
        console.print("[red bold]✘  Workflow completed with failures.[/red bold]")

    # Per-agent cost is persisted to costs_<thread_id>.json at run end (see
    # cli._emit_cost_report); no cost table is printed to the terminal.

    for name, status in statuses.items():
        # Successful runs: status table only — no logs/code clutter. Logs and code
        # are shown only for failures, to aid diagnosis.
        if status not in (FunctionStatus.FAILED, FunctionStatus.TIMEOUT):
            continue

        panels = []
        try:
            logs = runner.get_function_logs_content(name)
            log_tail = "\n".join(logs.splitlines()[-30:]) if logs else ""
        except Exception:
            log_tail = ""

        if log_tail:
            panels.append(Panel(log_tail, title=f"Failure Logs: {name}", border_style="red", expand=True))
        code = code_by_action.get(extract_function_name(name), "")
        if code:
            try:
                panels.append(Panel(
                    Syntax(code, "python", theme="monokai", line_numbers=True),
                    title=f"Code: {name}",
                    border_style="yellow",
                    expand=True,
                ))
            except Exception:
                pass

        if panels:
            console.print(Group(*panels))

    console.print()
    input("Press Enter to continue...")

    log_parts = []
    for fn_name, fn_obj in runner._functions.items():
        if fn_obj.logs_content:
            log_parts.append(f"=== {fn_name} ===\n{fn_obj.logs_content}")
    logs = "\n\n".join(log_parts)

    return {
        "stage": "monitor",
        "success": all_ok,
        "logs": logs,
        "errors": f"Failed: {failed_functions}" if failed_functions else "",
        "failed_functions": failed_functions,
        "invocation_id": invocation_id,
    }
