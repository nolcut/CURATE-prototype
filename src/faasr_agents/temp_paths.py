"""Temporary directory selection with a project-local fallback."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def usable_temp_dir() -> str:
    """Return a writable temp directory, creating a local fallback if needed."""
    candidates: list[Path] = []

    env_dir = os.environ.get("FAASR_TMPDIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    try:
        candidates.append(Path(tempfile.gettempdir()))
    except Exception:
        pass

    candidates.append(Path.cwd() / ".tmp")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=candidate, delete=True):
                pass
            return str(candidate)
        except OSError:
            continue

    raise FileNotFoundError(
        "No writable temporary directory found. Free disk space or set "
        "FAASR_TMPDIR to a writable folder."
    )
