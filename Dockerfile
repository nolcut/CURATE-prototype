FROM node:22-bookworm-slim

# Build toolchain + headers: the FGA agent pip-installs the generated functions'
# third-party imports and stub-tests them inside this container (agents/fga.py
# _SHARED_RULES), and a scientific package without a wheel for this arch compiles
# from source. nano backs the CLI's $EDITOR workflow-description prompt.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates pkg-config libgomp1 nano \
    && rm -rf /var/lib/apt/lists/*

# claude-agent-sdk spawns this CLI as a subprocess (src/faasr_agents/agents/fga.py).
# Pinned: `latest` makes builds irreproducible and can drift past the version the
# SDK checks for on connect.
RUN npm install -g @anthropic-ai/claude-code@2.1.220

# uv as a static binary (the Node base has no Python); uv then provisions CPython 3.13
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

WORKDIR /app

# /usr/local/bin precedes the venv so the credential-preflight wrapper installed there
# shadows the `faasr-agents` console script; python3/pip still come from the venv.
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    EDITOR=nano \
    MPLBACKEND=Agg \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/usr/local/bin:/app/.venv/bin:$PATH"

# uv sync (not pip) so pyproject's [tool.uv] boto3 override applies -- FaaSr_py pins a
# boto3 that predates Bedrock API-key auth. Split in two so editing src/ doesn't
# re-resolve and reinstall every dependency.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Editable install keeps catalog/ writable.
COPY src ./src
RUN uv sync --frozen

# A uv venv ships no pip, but the FGA agent is instructed to `pip install` the packages
# the generated functions import before stub-testing them. Seed one -- and do it AFTER
# the last `uv sync`, which prunes anything absent from the lockfile (pip included).
RUN uv pip install --python /app/.venv/bin/python pip setuptools wheel

# A login shell re-derives PATH from /etc/profile and would drop the venv, so `bash -lc`
# (and anything the FGA agent runs that way) would lose python3/pip. Re-prepend it there.
RUN printf 'PATH="/usr/local/bin:/app/.venv/bin:$PATH"\n' > /etc/profile.d/faasr-venv.sh

# setup_env.py writes ROOT/.env from ROOT/.env.template, where ROOT is its own
# directory -- at /app that lands on the /app/.env the CLI's load_dotenv() reads.
COPY setup_env.py .env.template ./

# Credential preflight: `faasr-agents` refuses to start unconfigured rather than
# failing deep inside an LLM, GitHub, or S3 call.
COPY docker/preflight.py /usr/local/share/curate/preflight.py
COPY docker/faasr-agents /usr/local/bin/faasr-agents
RUN chmod +x /usr/local/bin/faasr-agents

RUN printf '#!/bin/sh\nexec /app/.venv/bin/python3 /app/setup_env.py "$@"\n' \
      > /usr/local/bin/curate-setup \
    && chmod +x /usr/local/bin/curate-setup

RUN printf '%s\n' '' \
    'echo "  curate — FaaSr Agent System"' \
    'echo "    1. curate-setup     # write /app/.env  (or pass --env-file)"' \
    'echo "    2. faasr-agents     # start the agent system"' \
    'echo' >> /root/.bashrc

# Run out of /work, not /app, so a single `-v "$PWD:/work"` puts faasr_output/ and
# downloads/ straight onto the host -- both resolve relative to the cwd. The CLI finds
# /app/.env regardless, since python-dotenv walks up from the installed package.
WORKDIR /work

# The image must not launch the agent system on its own -- a fresh container has no
# credentials yet. Land in a shell, configure, then run `faasr-agents`.
# ENTRYPOINT is reset explicitly: leaving it unset inherits the node base image's
# docker-entrypoint.sh, which silently execs `node "$@"` for any argument starting
# with a dash (so `docker run IMAGE --opus` would run node, not the CLI).
ENTRYPOINT []
CMD ["bash"]
