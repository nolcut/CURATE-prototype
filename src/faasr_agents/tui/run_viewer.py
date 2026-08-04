"""
Scrollable TUI viewer for Gate-5 post-run code and logs.

run_output_viewer(code_by_node, logs) opens a Rich Live full-screen browser
and blocks until the user presses q or Enter, then returns so the REPL
can continue capturing decisions (accept / save / request).

Controls: ← → (or n/p) cycle items · ↑ ↓ scroll · q / Enter exit
"""
from __future__ import annotations

import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from faasr_agents.tui.code_review import _CodeView, _keyboard_listener

_LOGS_SENTINEL = "__logs__"
_WORKFLOW_SENTINEL = "__workflow__"
_SPECIAL_LABEL = {_LOGS_SENTINEL: "logs", _WORKFLOW_SENTINEL: "workflow.json"}


def _build_item_table(item_keys: list[str], viewed_idx: int) -> Table:
    table = Table(expand=True, show_header=False, show_lines=False, box=None, padding=(0, 1))
    table.add_column("", width=2)
    table.add_column("Name", ratio=1)
    for i, key in enumerate(item_keys):
        indicator = "▶" if i == viewed_idx else " "
        is_current = i == viewed_idx
        special = key in _SPECIAL_LABEL
        display = _SPECIAL_LABEL.get(key, key)
        if special:
            style = "bold italic" if is_current else "italic"
        else:
            style = "bold" if is_current else ""
        table.add_row(indicator, Text(display, style=style))
    return table


def _build_content_panel(
    key: str,
    code_by_node: dict[str, str],
    logs: str,
    workflow_json: str,
    view: _CodeView,
) -> Panel:
    if key == _LOGS_SENTINEL:
        content = logs or "(no logs)"
        language = "text"
        title_name = "execution logs"
    elif key == _WORKFLOW_SENTINEL:
        content = workflow_json or "(no workflow.json)"
        language = "json"
        title_name = "workflow.json"
    else:
        content = code_by_node.get(key) or "(no code)"
        language = "python"
        title_name = key

    all_lines = content.splitlines()
    total = len(all_lines)
    offset = view.get_offset(key)
    page = view.page_size
    visible = all_lines[offset: offset + page]
    top = offset + 1
    bot = min(offset + page, total)
    scroll_hint = f"lines {top}–{bot}/{total}" if total > page else f"{total} lines"
    title = f"[bold]{title_name}[/bold]   {scroll_hint}"

    if content not in ("(no logs)", "(no code)", "(no workflow.json)"):
        try:
            renderable = Syntax(
                "\n".join(visible),
                language,
                theme="monokai",
                line_numbers=True,
                start_line=offset + 1,
            )
        except Exception:
            renderable = Text("\n".join(visible))
    else:
        renderable = Text(content, style="dim italic")

    return Panel(renderable, title=title, border_style="green", expand=True)


def run_output_viewer(code_by_node: dict[str, str], logs: str, workflow_json: str = "") -> None:
    """Blocking Rich TUI: browse generated code, logs, and workflow.json. Returns on q/Enter."""
    item_keys = list(code_by_node.keys())
    if logs:
        item_keys.append(_LOGS_SENTINEL)
    if workflow_json:
        item_keys.append(_WORKFLOW_SENTINEL)
    if not item_keys:
        return

    view = _CodeView(item_keys)
    stop_event = threading.Event()

    def total_lines_for(name: str | None) -> int:
        if not name:
            return 0
        if name == _LOGS_SENTINEL:
            return len((logs or "").splitlines())
        if name == _WORKFLOW_SENTINEL:
            return len((workflow_json or "").splitlines())
        return len((code_by_node.get(name) or "").splitlines())

    kb_thread = threading.Thread(
        target=_keyboard_listener,
        args=(view, stop_event, total_lines_for),
        daemon=True,
    )
    kb_thread.start()

    layout = Layout()
    layout.split_column(
        Layout(name="main", ratio=1),
        Layout(name="footer", size=1),
    )
    layout["main"].split_row(
        Layout(name="list", size=26),
        Layout(name="content"),
    )

    _FOOTER = Text(
        "  ← →  cycle items    ↑ ↓  scroll    q / Enter  exit  ",
        justify="center",
        style="bold dim",
    )

    console = Console()
    try:
        with Live(layout, console=console, refresh_per_second=4, screen=True):
            while not stop_event.is_set():
                key = view.current or item_keys[0]
                view.page_size = max(10, console.height - 6)

                layout["list"].update(Panel(
                    _build_item_table(item_keys, view.index),
                    title=f"[bold]Items ({len(item_keys)})[/bold]",
                    border_style="cyan",
                ))
                layout["content"].update(
                    _build_content_panel(key, code_by_node, logs, workflow_json, view)
                )
                layout["footer"].update(_FOOTER)
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
