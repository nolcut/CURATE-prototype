"""
Beautiful Rich trace for the FGA Claude Agent SDK sessions (--debug only).

AgentTracer renders each Agent SDK stream message as it arrives — full thinking,
full assistant text, color-coded tool calls, tool errors — and simultaneously
accumulates a markdown transcript that fga.py saves under the kept debug context
dir (<ctxdir>/_trace/<fn>.md).

Nothing here is truncated: the whole point is to see exactly what the agent did.
"""
from __future__ import annotations

import shutil

from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from claude_agent_sdk import (
    AssistantMessage,
    UserMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

# tool name -> (rich style, icon)
_TOOL_STYLE: dict[str, tuple[str, str]] = {
    "Read":  ("green",   "▸"),
    "Glob":  ("green",   "▸"),
    "Grep":  ("green",   "▸"),
    "Write": ("yellow",  "✎"),
    "Edit":  ("yellow",  "✎"),
    "Bash":  ("magenta", "❯"),
}


import re


def _longest_backtick_run(s: str) -> int:
    """Length of the longest consecutive backtick run in s (0 if none)."""
    return max((len(m) for m in re.findall(r"`+", s)), default=0)


def _inline_code(s: str) -> str:
    """Wrap s as a backtick-safe inline code span (CommonMark rules).

    Uses a fence of N+1 backticks (N = longest run inside s) and pads with a
    single space on each side when s starts/ends with a backtick, so args
    containing backticks (e.g. grep patterns) don't break the span.
    """
    ticks = "`" * (_longest_backtick_run(s) + 1)
    pad = " " if (s.startswith("`") or s.endswith("`")) else ""
    return f"{ticks}{pad}{s}{pad}{ticks}"


def _fenced(code: str, lang: str = "") -> str:
    """Return code as a top-level fenced block, using a fence long enough to
    survive any backtick run inside code (min 3)."""
    fence = "`" * max(3, _longest_backtick_run(code) + 1)
    return f"{fence}{lang}\n{code}\n{fence}"


def _short_path(p: str) -> str:
    """Trim a long absolute path to its trailing components for readable traces."""
    parts = p.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else p


def _tool_arg(block: ToolUseBlock) -> str:
    """Extract the most informative argument for a tool call (no truncation)."""
    inp = block.input or {}
    if block.name in ("Read", "Write", "Edit"):
        return _short_path(str(inp.get("file_path", "?")))
    if block.name in ("Glob", "Grep"):
        return str(inp.get("pattern", "?"))
    if block.name == "Bash":
        return str(inp.get("command", "") or "")
    return ", ".join(f"{k}={v}" for k, v in inp.items())


class AgentTracer:
    """Renders an Agent SDK message stream and builds a markdown transcript."""

    def __init__(self, fn_name: str, source: str = "new"):
        self.fn_name = fn_name
        self.source = source
        # Cap render width so thinking panels don't stretch across a wide terminal.
        width = min(shutil.get_terminal_size((100, 40)).columns, 100)
        self.console = Console(width=width)
        self._md: list[str] = []

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self.console.rule(f"[bold]FGA · {self.fn_name}[/bold] [dim]({self.source})[/dim]")
        self._md.append(f"# FGA trace — {_inline_code(self.fn_name)} ({self.source})")

    def finish(self, result: ResultMessage | None) -> None:
        if result is None:
            return
        secs = (result.duration_ms or 0) / 1000.0
        cost = result.total_cost_usd or 0.0
        summary = f"✓ {result.num_turns} turns · {secs:.1f}s · ${cost:.4f}"
        self.console.print(f"  [dim]{summary}[/dim]")
        self._md.append(f"---\n\n_{summary}_")

    def transcript(self) -> str:
        # One blank line between every block so paragraphs, lists, code fences,
        # and blockquotes render as distinct elements in a markdown viewer.
        return "\n\n".join(self._md) + "\n"

    # ── per-message dispatch ───────────────────────────────────────────────
    def handle(self, msg) -> None:
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, ThinkingBlock):
                    self._thinking(b.thinking)
                elif isinstance(b, TextBlock):
                    self._text(b.text)
                elif isinstance(b, ToolUseBlock):
                    self._tool_use(b)
        elif isinstance(msg, UserMessage):
            content = msg.content if isinstance(msg.content, list) else []
            for b in content:
                if isinstance(b, ToolResultBlock) and b.is_error:
                    self._tool_error(b)
        elif isinstance(msg, ResultMessage):
            # finish() handles the summary explicitly; ignore here.
            pass

    # ── renderers ──────────────────────────────────────────────────────────
    def _thinking(self, thinking: str) -> None:
        text = (thinking or "").strip()
        if not text:
            return
        self.console.print(Padding(Panel(
            Text(text, style="italic dim"),
            title="thinking",
            title_align="left",
            border_style="grey37",
            padding=(0, 1),
        ), (0, 0, 0, 2)))
        self._md.append("> **thinking**\n>\n" + "\n".join(f"> {ln}" for ln in text.splitlines()))

    def _text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        body = Text("» ", style="bold cyan")
        body.append(text, style="cyan")
        self.console.print(Padding(body, (0, 0, 0, 2)))
        self._md.append(text)

    def _tool_use(self, block: ToolUseBlock) -> None:
        style, icon = _TOOL_STYLE.get(block.name, ("white", "•"))
        arg = _tool_arg(block)
        if block.name == "Bash":
            line = Text(f"  {icon} ", style=style)
            line.append("Bash", style=f"bold {style}")
            self.console.print(line)
            # full command, multi-line ok — indented under the Bash line
            self.console.print(Padding(Text(arg, style="dim"), (0, 0, 0, 4)))
            # Top-level fenced block (not nested in a list item — that breaks
            # CommonMark) so multi-line commands render as a code block.
            self._md.append("**Bash**\n\n" + _fenced(arg, "sh"))
        else:
            line = Text(f"  {icon} ", style=style)
            line.append(f"{block.name} ", style=f"bold {style}")
            line.append(arg, style="default")
            self.console.print(line)
            self._md.append(f"- **{block.name}** {_inline_code(arg)}")

    def _tool_error(self, block: ToolResultBlock) -> None:
        content = block.content
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
            )
        content = str(content).strip()
        self.console.print(Text(f"  ✗ error  {content}", style="red"))
        # Fenced so multi-line error text stays intact as one block.
        self._md.append("**✗ error**\n\n" + _fenced(content))
