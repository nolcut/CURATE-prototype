from __future__ import annotations
import os
import re
from faasr_agents.models import WorkflowSpec, DataFlowEdge, clean_dependencies


def _to_action_name(python_name: str) -> str:
    """Convert snake_case Python name to kebab-case FaaSr action name.
    FaaSr ActionList keys must match ^[a-zA-Z][a-zA-Z0-9-]*$
    """
    kebab = python_name.replace("_", "-")
    kebab = re.sub(r'[^a-zA-Z0-9-]', '-', kebab)
    kebab = re.sub(r'-+', '-', kebab).strip('-')
    if not kebab or not kebab[0].isalpha():
        kebab = 'fn-' + kebab
    return kebab


def _sanitize_workflow_name(name: str) -> str:
    """FaaSr WorkflowName must match ^[a-zA-Z][a-zA-Z0-9-]*$"""
    sanitized = re.sub(r'[^a-zA-Z0-9-]', '-', name)
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    if not sanitized or not sanitized[0].isalpha():
        sanitized = 'workflow-' + sanitized
    return sanitized


def external_inputs(spec: WorkflowSpec) -> list[dict]:
    """Inputs that no node produces — files the user must place in S3 beforehand.

    Returns one entry per distinct external file:
    {file, node, s3_path, type, description}. The description prefers the input's
    own IOSpec description and falls back to the consuming node's description
    (which already explains what the file is), so the user knows what to provide.
    """
    # Match produced files in a {rank}-tolerant way: a templated input
    # (shuffle_shard_{rank}.json) must be recognised as produced even when the
    # producer records concrete shard names (shuffle_shard_1.json, ...) and
    # vice-versa. Only {rank} is widened to \d+ — no blanket digit-collapsing —
    # so unrelated numbered files aren't falsely merged.
    def _base(n: str) -> str:
        return n.rsplit("/", 1)[-1]

    def _rank_pattern(base: str) -> re.Pattern:
        return re.compile("^" + re.escape(base).replace(r"\{rank\}", r"\d+") + "$")

    # Attribute production to the node that outputs it. A file counts as produced —
    # and so NOT external — only when some node OTHER than the consumer outputs it:
    # a node that lists a file as both input and output (validate-and-forward) must
    # not mask its own input, since that input is still supplied externally.
    produced_by: dict[str, set[str]] = {}
    patterns_by: list[tuple[re.Pattern, str]] = []
    for node in spec.nodes:
        for o in node.outputs:
            b = _base(o.name)
            produced_by.setdefault(b, set()).add(node.name)
            if "{rank}" in b:
                patterns_by.append((_rank_pattern(b), node.name))

    def _produced_by_other(name: str, consumer: str) -> bool:
        b = _base(name)
        if any(n != consumer for n in produced_by.get(b, ())):
            return True
        if any(prod != consumer and pat.match(b) for pat, prod in patterns_by):
            return True
        # A templated input is satisfied by any concrete produced shard from another node.
        if "{rank}" in b:
            pat = _rank_pattern(b)
            if any(
                prod != consumer and pat.match(lit)
                for lit, prods in produced_by.items()
                for prod in prods
            ):
                return True
        return False

    out: list[dict] = []
    seen: set[str] = set()
    for node in spec.nodes:
        for inp in node.inputs:
            if not _produced_by_other(inp.name, node.name) and inp.name not in seen:
                seen.add(inp.name)
                desc = (inp.description or "").strip() or (node.description or "").strip()
                out.append({
                    "file": inp.name,
                    "node": node.name,
                    "s3_path": f"{spec.folder}/{inp.name}" if spec.folder else inp.name,
                    "type": (inp.type or "").strip(),
                    "description": desc,
                })
    return out


def emit_faasr_json(spec: WorkflowSpec) -> dict:
    """Convert a WorkflowSpec into a FaaSr workflow JSON dict.

    ActionList keys are kebab-case (FaaSr schema constraint); FunctionName
    values are the original snake_case Python function names.
    """
    action_list = {}
    function_git_repo = {}

    # Map python_name → action_name (kebab) for InvokeNext references
    action_name = {node.name: _to_action_name(node.name) for node in spec.nodes}
    node_rank = {node.name: max(1, int(node.rank or 1)) for node in spec.nodes}

    def _invoke_target(to_node: str) -> str:
        # A ranked target is invoked as "name(N)" → N parallel instances.
        target = action_name[to_node]
        n = node_rank.get(to_node, 1)
        return f"{target}({n})" if n > 1 else target

    # Build InvokeNext map keyed by python_name, values are action_names
    invoke_next: dict[str, list] = {node.name: [] for node in spec.nodes}
    for edge in spec.edges:
        if edge.condition is None:
            invoke_next[edge.from_node].append(_invoke_target(edge.to_node))
        else:
            existing = invoke_next[edge.from_node]
            cond_entry = next((e for e in existing if isinstance(e, dict)), None)
            if cond_entry is None:
                cond_entry = {}
                existing.append(cond_entry)
            cond_entry.setdefault(edge.condition, []).append(_invoke_target(edge.to_node))

    gh_user = os.environ.get("FAASR_GH_USERNAME", "username")
    gh_repo = os.environ.get("FAASR_ACTION_REPO", "faasr-functions")
    default_container = os.environ.get(
        "FAASR_DEFAULT_CONTAINER", "ghcr.io/faasr/github-actions-python:latest"
    )

    for node in spec.nodes:
        aname = action_name[node.name]
        args: dict[str, str] = {"folder": spec.folder}
        for i, inp in enumerate(node.inputs, 1):
            args[f"input{i}"] = inp.name
        for i, out in enumerate(node.outputs, 1):
            args[f"output{i}"] = out.name

        # Always emit InvokeNext (empty list for terminal nodes). The FaaSr schema
        # doesn't require it, but FaaSr_py's check_dag/build_adjacency_graph reads it
        # unconditionally, and real FaaSr payloads carry it on every action.
        action: dict = {
            "FaaSServer": spec.compute_target,
            "Type": "Python",
            "FunctionName": node.name,  # Python function name (may have underscores)
            "Arguments": args,
            "InvokeNext": invoke_next.get(node.name, []),
        }

        action_list[aname] = action
        # FaaSr runtime resolves .py paths via faasr_get_github_raw, which
        # expects "owner/repo/branch/path/to/file.py" (≥4 parts).
        function_git_repo[node.name] = (
            f"{gh_user}/{gh_repo}/main/functions/{node.name}.py"
        )

    entry_python = spec.entry or (spec.nodes[0].name if spec.nodes else "")
    entry_action = action_name.get(entry_python, _to_action_name(entry_python))

    compute_servers = {
        spec.compute_target: {
            "FaaSType": "GitHubActions",
            "UserName": gh_user,
            "ActionRepoName": gh_repo,
            "Branch": "main",
            "UseSecretStore": True,
        }
    }
    data_stores = {
        spec.data_store: {
            "Endpoint": os.environ.get("FAASR_S3_ENDPOINT", "https://s3.amazonaws.com"),
            "Bucket": os.environ.get("FAASR_S3_BUCKET", "faasr-data"),
            "Region": os.environ.get("FAASR_S3_REGION", "us-east-1"),
            # NEVER embed real credentials in the payload: it is committed to GitHub
            # (leaking secrets + tripping push protection). FaaSr's convention is to
            # store placeholder secret-names here; the real values are injected at
            # runtime from GitHub Actions secrets (named <DataStore>_AccessKey /
            # <DataStore>_SecretKey by register_workflow) or from local env at monitor time.
            "AccessKey": f"{spec.data_store}_AccessKey",
            "SecretKey": f"{spec.data_store}_SecretKey",
            "Writable": "TRUE",
        }
    }

    # Every action runs in the default FaaSr Python container unless overridden.
    action_containers = {aname: default_container for aname in action_list}

    # Third-party PyPI packages only — never the faasr runtime or stdlib modules.
    pypi_packages = {}
    for node in spec.nodes:
        deps = clean_dependencies(node.dependencies)
        if deps:
            pypi_packages[node.name] = deps

    result: dict = {
        "FunctionInvoke": entry_action,
        "WorkflowName": _sanitize_workflow_name(spec.name),
        "DefaultDataStore": spec.data_store,
        "LoggingDataStore": spec.data_store,
        "FaaSrLog": "FaaSrLog",
        "InvocationIDFromDate": "%Y-%m-%d-%H-%M-%S",
        "ActionList": action_list,
        "ActionContainers": action_containers,
        "ComputeServers": compute_servers,
        "DataStores": data_stores,
        "FunctionGitRepo": function_git_repo,
    }
    if pypi_packages:
        result["PyPIPackageDownloads"] = pypi_packages

    # Workflow-wide secrets = union of per-function secrets. register_workflow.py
    # reads this top-level list to inject `NAME: ${{ secrets.NAME }}` into the
    # GitHub Actions env block of every action.
    secrets = sorted({s for node in spec.nodes for s in node.secrets})
    if secrets:
        result["Secrets"] = secrets
    return result
