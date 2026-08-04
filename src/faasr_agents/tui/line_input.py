"""
Paste-aware single-line input for interactive prompts (e.g. the Gate 5 REPL).

`read_line(prompt)` behaves like `input(prompt)` for typed text, but enables terminal
**bracketed paste** so a multiline paste is captured as ONE value and shown inline as a
compact placeholder (`[pasted N chars]`) instead of flooding the prompt / being split at
the first newline (which plain `input()` does).

Falls back to `input(prompt)` when stdin/stdout isn't a TTY or the platform lacks termios
(e.g. Windows, pipes, CI) — so non-interactive runs and tests are unaffected.

The buffer/segment logic (`_LineBuffer`) and the bracketed-paste markers are kept free of
any terminal I/O so they can be unit-tested directly.
"""
from __future__ import annotations

import sys

# Bracketed-paste control sequences (see xterm "bracketed paste mode").
ENABLE_BRACKETED_PASTE = "\x1b[?2004h"
DISABLE_BRACKETED_PASTE = "\x1b[?2004l"
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"

# Dim styling for the inline placeholder (matches the CLI's DIM helper).
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _paste_placeholder(text: str) -> str:
    """Human-readable summary shown in place of a pasted block."""
    n = len(text)
    lines = text.count("\n") + 1
    if lines > 1:
        return f"[pasted {n} chars, {lines} lines]"
    return f"[pasted {n} chars]"


class _LineBuffer:
    """A single logical input line built from typed characters and pasted blocks.

    Typed characters are stored verbatim; each paste is stored as one indivisible
    segment so it renders as a compact placeholder and deletes as a unit. `value()`
    returns the fully-expanded text; `render()` returns what the user sees.
    """

    def __init__(self) -> None:
        # Ordered list of segments: ("text", str) for typed chars, ("paste", str) for pastes.
        self._segments: list[tuple[str, str]] = []

    def insert_text(self, s: str) -> None:
        """Append typed character(s). Coalesces into the trailing text segment."""
        if not s:
            return
        if self._segments and self._segments[-1][0] == "text":
            kind, cur = self._segments[-1]
            self._segments[-1] = ("text", cur + s)
        else:
            self._segments.append(("text", s))

    def insert_paste(self, s: str) -> None:
        """Append a pasted block as one indivisible segment."""
        self._segments.append(("paste", s))

    def backspace(self) -> None:
        """Delete one unit from the end: one char from trailing text, or a whole paste."""
        if not self._segments:
            return
        kind, val = self._segments[-1]
        if kind == "paste":
            self._segments.pop()
        else:
            if len(val) <= 1:
                self._segments.pop()
            else:
                self._segments[-1] = ("text", val[:-1])

    def value(self) -> str:
        """The fully-expanded text (pastes inlined verbatim)."""
        return "".join(val for _, val in self._segments)

    def render(self) -> str:
        """What the user sees: typed text verbatim, pastes as dim placeholders."""
        out = []
        for kind, val in self._segments:
            if kind == "paste":
                out.append(f"{_DIM}{_paste_placeholder(val)}{_RESET}")
            else:
                out.append(val)
        return "".join(out)

    def is_empty(self) -> bool:
        return not self._segments


def read_line(prompt: str = "") -> str:
    """Read one line of input with bracketed-paste support.

    Returns the fully-expanded text (multiline pastes included). Raises KeyboardInterrupt
    on Ctrl-C and EOFError on Ctrl-D at an empty buffer, matching `input()` semantics that
    the REPL already handles. Falls back to plain `input()` when not on a capable TTY.
    """
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except Exception:
        return input(prompt)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input(prompt)

    return _read_line_raw(prompt)


def _read_line_raw(prompt: str) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf = _LineBuffer()

    def redraw() -> None:
        # \r → col 0, \x1b[K → clear to EOL, then prompt + current rendering.
        sys.stdout.write("\r\x1b[K" + prompt + buf.render())
        sys.stdout.flush()

    sys.stdout.write(ENABLE_BRACKETED_PASTE)
    sys.stdout.flush()
    try:
        tty.setcbreak(fd)
        redraw()
        pending = ""  # unconsumed bytes already read (paste content can arrive in one chunk)
        while True:
            ch = pending or sys.stdin.read(1)
            pending = ""
            if ch == "":
                # EOF from the stream
                if buf.is_empty():
                    raise EOFError
                break
            if ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
            if ch == "\x04":  # Ctrl-D
                if buf.is_empty():
                    raise EOFError
                continue
            if ch in ("\r", "\n"):
                break
            if ch in ("\x7f", "\b"):  # Backspace / DEL
                buf.backspace()
                redraw()
                continue
            if ch == "\x1b":  # escape sequence — could be bracketed paste or an arrow
                seq = _read_escape_sequence()
                if seq == "paste_start":
                    pasted = _read_until(PASTE_END)
                    buf.insert_paste(pasted)
                    redraw()
                # arrows / other CSI sequences are ignored (cursor stays at end)
                continue
            if ch < " " and ch != "\t":  # other control chars — ignore
                continue
            buf.insert_text(ch)
            redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(DISABLE_BRACKETED_PASTE + "\n")
        sys.stdout.flush()

    return buf.value()


def _read_escape_sequence() -> str:
    """After an ESC byte, read the rest of a CSI sequence. Returns 'paste_start' for the
    bracketed-paste-begin marker, else 'other'."""
    b = sys.stdin.read(1)
    if b != "[":
        return "other"
    # Read the numeric/terminator body: PASTE_START is ESC [ 2 0 0 ~
    body = ""
    while True:
        c = sys.stdin.read(1)
        if c == "" or not (c.isdigit()):
            body += c
            break
        body += c
    return "paste_start" if body == "200~" else "other"


def _read_until(terminator: str) -> str:
    """Read chars until `terminator` is seen; return the content before it."""
    out = ""
    while True:
        c = sys.stdin.read(1)
        if c == "":
            break
        out += c
        if out.endswith(terminator):
            return out[: -len(terminator)]
    return out
