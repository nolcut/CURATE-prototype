#!/usr/bin/env python3
"""Refuse to launch the agent system when credentials are missing.

Runs ahead of the real `faasr-agents` console script inside the container (see
docker/faasr-agents), so an unconfigured run fails immediately with an actionable
message instead of dying deep inside an LLM, GitHub, or S3 call.

The same .env the CLI will load is resolved here too, so values sitting in /app/.env
count as configured -- the check agrees with what the CLI actually sees.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _cli_dotenv() -> Path | None:
    """Resolve the .env that cli.py's bare load_dotenv() will pick up.

    python-dotenv walks up from the *calling file's* directory, and cli.py is inside
    the installed package (/app/src/faasr_agents/cli.py), so it lands on /app/.env.
    This script lives elsewhere (/usr/local/share/curate), so calling load_dotenv()
    here would search the wrong tree and report a configured setup as missing.
    Reproduce the CLI's walk instead of guessing.
    """
    import faasr_agents

    start = Path(faasr_agents.__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


_dotenv = _cli_dotenv()
if _dotenv is not None:
    load_dotenv(_dotenv)

BOLD = lambda t: f"\033[1m{t}\033[0m"      # noqa: E731
DIM = lambda t: f"\033[2m{t}\033[0m"        # noqa: E731
RED = lambda t: f"\033[91m{t}\033[0m"       # noqa: E731
CYAN = lambda t: f"\033[96m{t}\033[0m"      # noqa: E731


def _set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _problems(argv: list[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    # LLM credentials. Bedrock is the default; --anthropic-api / --openai-api
    # opt into direct provider APIs and are the only cases where those keys are used.
    if "--anthropic-api" in argv and "--openai-api" in argv:
        found.append(("LLM_PROVIDER", "--anthropic-api and --openai-api are mutually exclusive"))
    elif "--anthropic-api" in argv:
        if not _set("ANTHROPIC_API_KEY"):
            found.append(("ANTHROPIC_API_KEY", "required by --anthropic-api"))
    elif "--openai-api" in argv:
        if not _set("OPENAI_API_KEY"):
            found.append(("OPENAI_API_KEY", "required by --openai-api"))
    elif not (
        _set("BEDROCK_API_KEY")
        or (_set("AWS_ACCESS_KEY_ID") and _set("AWS_SECRET_ACCESS_KEY"))
        or _set("AWS_PROFILE")
    ):
        found.append((
            "BEDROCK_API_KEY",
            "or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or AWS_PROFILE",
        ))

    for name in ("GH_PAT", "FAASR_GH_USERNAME", "FAASR_ACTION_REPO"):
        if not _set(name):
            found.append((name, "pushing functions / triggering GitHub Actions"))

    for name in ("FAASR_S3_BUCKET", "S3_AccessKey", "S3_SecretKey"):
        if not _set(name):
            found.append((name, "reading and writing workflow data"))

    return found


def main() -> int:
    if _set("FAASR_SKIP_PREFLIGHT"):
        return 0

    problems = _problems(sys.argv[1:])
    if not problems:
        return 0

    width = max(len(name) for name, _ in problems)
    print()
    print(f"  {RED('✘')}  {BOLD('Not configured — refusing to start.')}")
    print()
    print(f"  {DIM('Missing:')}")
    for name, why in problems:
        print(f"    {BOLD(name.ljust(width))}  {DIM(why)}")
    print()
    print(f"  {DIM('Configure either way:')}")
    print(f"    {CYAN('python3 setup_env.py')}"
          f"   {DIM('— fill in /app/.env interactively (this container only)')}")
    print(f"    {CYAN('docker run --env-file .env ...')}"
          f"   {DIM('— pass a host .env in instead')}")
    print()
    print(f"  {DIM('FAASR_SKIP_PREFLIGHT=1 bypasses this check.')}")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
