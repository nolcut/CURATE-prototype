"""
Interactive code-review browser for FaaSr-generated Python functions.

Launched at Gate 3 (after FGA) so the user can browse syntax-highlighted code
before approving or requesting changes.  Browse-only: the approve/change decision
is captured by the normal cli.py prompt after this returns.

Controls: ← → (or n/p) cycle functions · ↑ ↓ scroll code · q/Enter exit
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

_CODE_PAGE_DEFAULT = 35  # fallback if console height is unavailable


class _CodeView:
    """Tracks which function is viewed; handles per-function scroll offsets."""

    def __init__(self, names: list[str]):
        self._names = names
        self._idx = 0
        self._offsets: dict[str, int] = {}
        self._lock = threading.Lock()
        self.page_size = _CODE_PAGE_DEFAULT  # updated each render cycle

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

    def prev(self) -> None:
        with self._lock:
            if self._names:
                self._idx = (self._idx - 1) % len(self._names)

    def scroll_up(self, amount: int = 5) -> None:
        with self._lock:
            name = self._names[self._idx] if self._names else None
            if name:
                self._offsets[name] = max(0, self._offsets.get(name, 0) - amount)

    def scroll_down(self, total_lines: int, amount: int = 5) -> None:
        with self._lock:
            name = self._names[self._idx] if self._names else None
            if name:
                max_off = max(0, total_lines - self.page_size)
                self._offsets[name] = min(max_off, self._offsets.get(name, 0) + amount)

    def get_offset(self, name: str) -> int:
        with self._lock:
            return self._offsets.get(name, 0)


def _keyboard_listener(
    view: _CodeView,
    stop: threading.Event,
    total_lines_fn,
) -> None:
    try:
        import readchar
    except ImportError:
        return
    try:
        while not stop.is_set():
            key = readchar.readkey()
            if key in (readchar.key.RIGHT, "n"):
                view.next()
            elif key in (readchar.key.LEFT, "p"):
                view.prev()
            elif key == readchar.key.UP:
                view.scroll_up()
            elif key == readchar.key.DOWN:
                total = total_lines_fn(view.current)
                view.scroll_down(total)
            elif key in ("q", "\r", "\n"):
                stop.set()
    except Exception:
        pass


def _build_fn_list_table(functions: list[dict], viewed_idx: int) -> Table:
    table = Table(
        expand=True,
        show_header=False,
        show_lines=False,
        box=None,
        padding=(0, 1),
    )
    table.add_column("", width=2)
    table.add_column("Name", ratio=1)
    table.add_column("Tags", width=10, justify="right")

    _SRC = {
        "catalog": ("green",  "[catalog]"),
        "cached":  ("cyan",    "[cached]"),
        "adapt":   ("blue",    "[adapt]"),
        "new":     ("yellow",  "[new]"),
        "json":    ("magenta", "[json]"),
    }

    for i, fn in enumerate(functions):
        indicator = "▶" if i == viewed_idx else " "
        source = fn.get("source", "new")
        tags = Text()
        if source == "user_provided":
            umm = fn.get("user_model_mode")
            tags.append("[user]", style="magenta")
            if umm:
                tags.append("\n")
                tags.append(f"[{umm}]", style="magenta")
        else:
            src_style, src_label = _SRC.get(source, ("dim", "[?]"))
            tags.append(src_label, style=src_style)
        table.add_row(
            indicator,
            Text(fn["name"], style="bold" if i == viewed_idx else ""),
            tags,
        )

    return table


def _build_code_panel(fn: dict, view: _CodeView) -> Panel:
    code = fn.get("code") or ""
    lines = code.splitlines()
    total = len(lines)
    offset = view.get_offset(fn["name"])
    page = view.page_size

    visible = lines[offset: offset + page]
    top = offset + 1
    bot = min(offset + page, total)

    scroll_hint = f"lines {top}–{bot}/{total}" if total > page else f"{total} lines"
    title = f"[bold]{fn['name']}[/bold]   {scroll_hint}"

    if code:
        try:
            renderable = Syntax(
                "\n".join(visible),
                fn.get("language", "python"),
                theme="monokai",
                line_numbers=True,
                start_line=offset + 1,
            )
        except Exception:
            renderable = Text("\n".join(visible))
    else:
        renderable = Text("(no code generated)", style="dim italic")

    return Panel(renderable, title=title, border_style="green", expand=True)


def run_code_review(functions: list[dict], workflow_json: str = "") -> None:
    """Display an interactive syntax-highlighted code browser.

    If workflow_json is given, it is browsable as an extra "workflow.json" item
    (rendered as JSON) alongside the functions.

    Blocks until the user presses q or Enter, then returns so cli.py can
    issue the approve/change prompt.
    """
    items = list(functions)
    if workflow_json:
        items.append({
            "name": "workflow.json",
            "code": workflow_json,
            "language": "json",
            "source": "json",
        })
    if not items:
        return

    names = [fn["name"] for fn in items]
    fn_by_name = {fn["name"]: fn for fn in items}

    view = _CodeView(names)
    stop_event = threading.Event()

    def total_lines_for(name: str | None) -> int:
        if not name:
            return 0
        return len((fn_by_name.get(name, {}).get("code") or "").splitlines())

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
        Layout(name="list", size=36),
        Layout(name="code"),
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
                fn_name = view.current
                fn = fn_by_name.get(fn_name or "") or items[0]

                # Dynamically fit code to terminal height: subtract panel borders + footer
                view.page_size = max(10, console.height - 6)

                layout["list"].update(Panel(
                    _build_fn_list_table(items, view.index),
                    title=f"[bold]Items ({len(items)})[/bold]",
                    border_style="cyan",
                ))
                layout["code"].update(_build_code_panel(fn, view))
                layout["footer"].update(_FOOTER)

                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
