from __future__ import annotations
import json
import re
import textwrap
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import interrupt
from faasr_agents.llm import get_llm
from faasr_agents.pricing import record_usage
from faasr_agents.models import (
    WorkflowSpec, FunctionSpec, WorkflowEdge, DataFlowEdge, IOSpec, clean_dependencies
)
from faasr_agents.state import AgentState
from faasr_agents.prompts.wca_prompts import (
    WCA_SYSTEM, WCA_SKELETON_PROMPT, WCA_RESOLVE_PROMPT, WCA_CHANGE_CLASSIFY_PROMPT
)
from faasr_agents.agents.fca import propose_candidate
from faasr_agents.faasr.emit import external_inputs


def _extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _accumulated_feedback(state) -> str:
    """All standing user change-request directives (accumulated across revision rounds),
    rendered as a bullet list. Includes the transient `feedback` (e.g. WDA auto-failure
    text) when it isn't already the last directive, so nothing is shown twice. This is
    what the skeleton/FCA/resolve prompts consume so a prior directive is never lost when
    a new one is issued."""
    directives = list(state.get("revision_directives") or [])
    fb = (state.get("feedback") or "").strip()
    if fb and (not directives or directives[-1] != fb):
        directives.append(fb)
    return "\n".join(f"- {d}" for d in directives)


def classify_structural_change(feedback: str, spec) -> bool:
    """Classify an explicit change request as STRUCTURE (topology) vs FUNCTION.

    Returns True when the change would alter the workflow's structure (add/remove/
    reorder nodes, rewire edges/dependencies, or change a node's parallelism/rank),
    so routing should go back to the skeleton stage. Returns False for changes that
    only affect the behavior/code/IO of existing functions (→ resolution stage).
    Defaults to False (the cheaper resolution path) when the request is empty or the
    model's answer is unclear. Recorded in the token tracker as "Revision Classification".
    """
    if not (feedback or "").strip() or spec is None:
        return False

    nodes = getattr(spec, "nodes", None) or []
    edges = getattr(spec, "edges", None) or []
    node_lines = "\n".join(
        f"  - {n.name}" + (f" (rank {n.rank})" if getattr(n, "rank", 1) and n.rank > 1 else "")
        for n in nodes
    ) or "  (none)"
    edge_lines = "\n".join(f"  {e.from_node} → {e.to_node}" for e in edges) or "  (none)"
    dag_summary = f"Nodes:\n{node_lines}\nEdges:\n{edge_lines}"

    prompt = WCA_CHANGE_CLASSIFY_PROMPT.format(dag_summary=dag_summary, feedback=feedback.strip())
    response = get_llm().invoke([
        SystemMessage(content=WCA_SYSTEM), HumanMessage(content=prompt)
    ])
    record_usage(response, "Revision Classification")
    return "STRUCTURE" in (response.content or "").strip().upper()



def _parse_workflow_spec(data: dict, skeleton: bool = False) -> WorkflowSpec:
    """Parse a raw dict into a WorkflowSpec, tolerating missing optional fields.

    When skeleton=True (Gate 1), nodes are kept as high-level natural-language outlines:
    inputs/outputs/dependencies are dropped even if the model volunteered them. These are
    made concrete in the resolve stage (Gate 2), aligned with catalog candidates.
    """
    nodes = []
    for n in data.get("nodes", []):
        if skeleton:
            inputs, outputs, dependencies = [], [], []
        else:
            inputs = [IOSpec(**i) if isinstance(i, dict) else IOSpec(name=str(i)) for i in n.get("inputs", [])]
            outputs = [IOSpec(**o) if isinstance(o, dict) else IOSpec(name=str(o)) for o in n.get("outputs", [])]
            dependencies = n.get("dependencies", [])
        nodes.append(FunctionSpec(
            name=n["name"],
            description=n.get("description", ""),
            inputs=inputs,
            outputs=outputs,
            dependencies=dependencies,
            rank=max(1, int(n.get("rank", 1) or 1)),  # >1 → ranked/parallel
        ))

    edges = [WorkflowEdge(**e) for e in data.get("edges", [])]

    data_flow = []
    for df in data.get("data_flow", []):
        data_flow.append(DataFlowEdge(**df))

    return WorkflowSpec(
        name=data.get("name", "workflow"),
        description=data.get("description", ""),
        nodes=nodes,
        edges=edges,
        data_flow=data_flow,
        entry=data.get("entry", nodes[0].name if nodes else ""),
        folder=data.get("folder", "workflow_data"),
        compute_target=data.get("compute_target", "GH"),
        # DataStore name is an infrastructure binding, not an LLM choice: it must
        # be "S3" so the payload's credential references (S3_AccessKey/S3_SecretKey),
        # the GitHub Actions secret names, and the local env vars all line up.
        data_store="S3",
    )


def _skeleton_outline(spec: WorkflowSpec) -> str:
    """Compact natural-language outline of an existing spec, used to seed a
    skeleton revision (so a reused/stored workflow can be revised in place)."""
    lines = []
    for n in spec.nodes:
        rank = f" rank={n.rank}" if (n.rank or 1) > 1 else ""
        lines.append(f"  - {n.name}{rank}: {(n.description or '').strip()}")
    edges = ", ".join(f"{e.from_node}->{e.to_node}" for e in spec.edges)
    body = "\n".join(lines)
    if edges:
        body += f"\n  edges: {edges}"
    return (
        "Current workflow being revised (apply the requested change; keep "
        "everything else identical):\n" + body + "\n"
    )


def _user_models_section(user_models) -> str:
    """Block listing user-attached scripts/functions that MUST appear as nodes."""
    if not user_models:
        return ""
    lines = [
        f"  - {m.name}: {(m.description or '(user-provided script/function)').strip()[:120]}"
        for m in user_models
    ]
    return (
        "User-provided functions (guaranteed steps — each MUST be a node with this EXACT "
        "snake_case name; the FGA will refactor the attached code into a FaaSr function):\n"
        + "\n".join(lines) + "\n"
    )


def _context_files_section(context_files) -> str:
    """Block listing user-attached reference files. These are NOT nodes — the FGA
    reads them from context/ for domain grounding; the planner just needs awareness."""
    if not context_files:
        return ""
    lines = [
        f"  - {c.name}" + (f": {c.description.strip()[:120]}" if c.description else "")
        for c in context_files
    ]
    return (
        "User-provided reference files (background material — papers, I/O examples, "
        "datasets; NOT nodes, do not add a node for them):\n"
        + "\n".join(lines) + "\n"
    )


def wca_compose_skeleton_node(state: AgentState) -> dict:
    """
    Compose (or revise) the workflow skeleton — the working half of Gate 1.

    Fresh entry (start of run, or a structural revision from a later gate):
    builds the skeleton prompt from the request + accumulated feedback and
    invokes the WCA. Re-entry from the skeleton gate with a pending refinement:
    continues the stored conversation instead of starting over. Either way the
    result is written to `skeleton_draft` for the gate node to present.
    """
    llm = get_llm()
    draft = state.get("skeleton_draft")

    if draft and draft.get("refinement"):
        # Continue the Gate-1 conversation with the user's refinement.
        messages = list(draft["conversation"])
        messages.append(AIMessage(content=draft["raw"]))
        messages.append(HumanMessage(content=(
            f"Please revise the workflow: {draft['refinement']}\n\n"
            "Respond with ONLY the complete workflow JSON object — no prose, no markdown."
        )))
        last_good_spec = draft.get("last_good")
    else:
        user_request = state["user_request"]
        accumulated = _accumulated_feedback(state)

        feedback_section = ""
        if accumulated:
            feedback_section = (
                "\nStanding change requests to incorporate (apply ALL — later items override "
                "earlier ones on conflict):\n" + accumulated + "\n"
            )

        # When revising an existing/reused workflow (e.g. a Gate-4 "I want N=2"
        # request), seed the prompt with the current spec so the change applies to
        # it. A reused workflow has no user_request, so fall back to a placeholder.
        existing = state.get("workflow_spec")
        current_workflow_section = ""
        if existing is not None and getattr(existing, "nodes", None):
            current_workflow_section = "\n" + _skeleton_outline(existing)
            if not (user_request or "").strip():
                user_request = "(revising the stored workflow shown below)"

        prompt = WCA_SKELETON_PROMPT.format(
            user_request=user_request,
            feedback_section=feedback_section,
            current_workflow_section=current_workflow_section,
            user_models_section=_user_models_section(state.get("user_models")),
            context_files_section=_context_files_section(state.get("context_files")),
        )
        messages = [SystemMessage(content=WCA_SYSTEM), HumanMessage(content=prompt)]
        last_good_spec = None

    response = llm.invoke(messages)
    # Registers into the run accumulator (faasr_agents.pricing). Each Gate-1
    # revision round fires exactly one real invoke.
    record_usage(response, "WCA")
    raw = response.content

    try:
        data = _extract_json(raw)
        # Prune here too: a skeleton revision is seeded with the current spec's
        # edges (_skeleton_outline), so a redundant edge would otherwise survive
        # the round-trip and show up in the Gate-1 DAG the user is reviewing.
        spec = _prune_transitive_edges(_parse_workflow_spec(data, skeleton=True))
        last_good_spec = spec
    except Exception as e:
        # A bad revision must not destroy the previously-approved skeleton.
        spec = last_good_spec
        if last_good_spec is not None:
            parse_error = (
                f"(Could not parse the revision: {e}. "
                "Showing the previous skeleton — approve it, or describe changes again.)"
            )
        else:
            parse_error = f"(Error parsing workflow: {e})"
    else:
        parse_error = None

    return {
        "skeleton_draft": {
            "conversation": messages,
            "raw": raw,
            "spec": spec,
            "parse_error": parse_error,
            "last_good": last_good_spec,
            "refinement": None,
        },
    }


def wca_skeleton_gate_node(state: AgentState) -> dict:
    """
    HITL Gate 1: present the composed skeleton and wait for approval/refinement.

    Approval commits the draft to workflow_spec (stashing the prior code-bearing
    spec) and clears the draft; anything else records a refinement on the draft
    and loops back to the compose node.
    """
    draft = state["skeleton_draft"]
    spec = draft["spec"]
    raw = draft["raw"]
    parse_error = draft["parse_error"]

    hitl_response = interrupt({
        "gate": 1,
        "message": "Review the proposed workflow skeleton. Type 'approve' to continue or describe changes."
            + (f"\n\n{parse_error}" if parse_error else ""),
        "workflow_spec": spec.model_dump() if spec else None,
    })

    approved = isinstance(hitl_response, str) and hitl_response.strip().lower() == "approve"
    if approved and spec is not None:
        result = {
            "workflow_spec": spec,
            "messages": [AIMessage(content=raw)],
            "hitl_gate": 1,
            "hitl_decision": "approve",
            "skeleton_draft": None,
            "spec_is_seed": None,
        }
        # Stash the prior code-bearing spec before the stripped skeleton
        # overwrites workflow_spec, so wca_fca can reuse this workflow's own
        # prior implementations verbatim (source="cached") on a revision.
        # Skipped when the spec only carries Gate-2 seeds (spec_is_seed): those
        # are catalog/adapt starting points the FGA never ran, so offering them
        # back as [cached] would launder a rejected seed into "prior work". Any
        # genuinely-implemented prior_spec from an earlier round stays put.
        existing = state.get("workflow_spec")
        if existing is not None and getattr(existing, "nodes", None) and not state.get("spec_is_seed"):
            result["prior_spec"] = existing
        return result

    if approved and spec is None:
        # Nothing valid to approve yet — ask the model to produce a skeleton.
        refinement = "Produce a valid workflow skeleton."
    else:
        refinement = hitl_response if isinstance(hitl_response, str) else str(hitl_response)
    return {
        "skeleton_draft": {**draft, "refinement": refinement},
        "hitl_gate": 1,
        "hitl_decision": "refine",
    }


def _fca_candidates_text(fca_summary: list[dict]) -> str:
    def _io(c):
        ins = ", ".join(c["inputs"]) or "—"
        outs = ", ".join(c["outputs"]) or "—"
        return f"inputs=[{ins}] outputs=[{outs}]"
    return "\n".join(
        f"  {c['name']}: [{c['source'].upper()}] — {c['description'][:80]}\n"
        f"      {_io(c)}"
        for c in fca_summary
    )


def _merge_resolved(raw: str, enriched_nodes: list) -> Any:
    """Parse a resolved WorkflowSpec JSON and merge catalog candidate I/O back in.

    source="catalog" → verbatim reuse: force-restore code/IO from catalog.
    source="cached"  → verbatim reuse of this workflow's OWN prior implementation:
                        force-restore code/IO from the prior node.
    source="adapt"        → seed-only: keep catalog code as FGA seed, let LLM-resolved
                        description/IO stand so revisions can change them.
    source="user_provided" → same as adapt, but the seed is the user's original
                        script/function; user_model_mode says whether the FGA wraps it
                        verbatim or refactors it (parity verified either way).
    source="new"     → fully LLM-resolved: no force-restore.
    """
    data = _extract_json(raw)
    resolved = _parse_workflow_spec(data)
    name_to_candidate = {n.name: n for n in enriched_nodes}
    final_nodes = []
    for node in resolved.nodes:
        candidate = name_to_candidate.get(node.name, node)
        if candidate.source in ("catalog", "cached"):
            update = {
                "source": candidate.source,
                "catalog_id": candidate.catalog_id,
                "origin_kind": candidate.origin_kind,
                "origin_name": candidate.origin_name,
                "code": candidate.code,
                "dependencies": clean_dependencies(candidate.dependencies or node.dependencies),
                "secrets": candidate.secrets,
                "inputs": candidate.inputs or node.inputs,
                "outputs": candidate.outputs or node.outputs,
            }
        elif candidate.source in ("adapt", "user_provided"):
            update = {
                "source": candidate.source,
                "user_model_mode": candidate.user_model_mode,
                "catalog_id": candidate.catalog_id,
                "origin_kind": candidate.origin_kind,
                "origin_name": candidate.origin_name,
                "code": candidate.code,   # seed for FGA agent (catalog item, or the user's original)
                "dependencies": clean_dependencies(candidate.dependencies or node.dependencies),
                "secrets": candidate.secrets,
                # inputs/outputs come from the LLM-resolved spec (node) so revisions can change them
            }
        else:
            update = {
                "source": candidate.source,
                "catalog_id": candidate.catalog_id,
                "code": candidate.code,
                "dependencies": clean_dependencies(candidate.dependencies or node.dependencies),
                "secrets": candidate.secrets,
            }
        final_nodes.append(node.model_copy(update=update))
    return resolved.model_copy(update={"nodes": final_nodes})


def _apply_candidate_renames(
    spec: WorkflowSpec, enriched_nodes: list[FunctionSpec]
) -> WorkflowSpec:
    """Propagate FCA node renames into the spec's edges, data_flow and entry.

    Verbatim catalog reuse keeps the catalog function's name — FaaSr's FunctionName
    is node.name (emit.py) and must match the `def` in the reused code — so a node
    can come back from the FCA under a different name than the one the skeleton's
    edges reference. Without rewriting those references the spec is left with
    dangling edges: _ensure_reachable then sees the renamed nodes as orphans and
    invents wrong trigger edges for them, and the stale names KeyError in the
    Gate-2 DAG display and in emit at deploy time.

    On a name collision (two nodes resolving to the same catalog function, or the
    catalog name already taken by another node) the candidate keeps its skeleton
    name and is downgraded to source="adapt", so the FGA regenerates it from the
    catalog seed as `def <skeleton_name>`.
    """
    orig_names = [n.name for n in spec.nodes]
    desired = [(c.name or o) for c, o in zip(enriched_nodes, orig_names)]
    renames: dict[str, str] = {}
    final_nodes: list[FunctionSpec] = []
    for i, (orig_name, cand) in enumerate(zip(orig_names, enriched_nodes)):
        new_name = desired[i]
        if new_name == orig_name:
            final_nodes.append(cand)
            continue
        collides = (
            new_name in orig_names[:i] + orig_names[i + 1:]
            or new_name in desired[:i] + desired[i + 1:]
        )
        if collides:
            final_nodes.append(cand.model_copy(update={"name": orig_name, "source": "adapt"}))
            print(
                f"  ◌  WCA  catalog name '{new_name}' collides with another node — "
                f"keeping '{orig_name}' as [adapt]",
                flush=True,
            )
            continue
        renames[orig_name] = new_name
        final_nodes.append(cand)

    if not renames:
        return spec.model_copy(update={"nodes": final_nodes})

    def _r(name: str) -> str:
        return renames.get(name, name)

    edges = [
        e.model_copy(update={"from_node": _r(e.from_node), "to_node": _r(e.to_node)})
        for e in spec.edges
    ]
    data_flow = [
        d.model_copy(update={"from_node": _r(d.from_node), "to_node": _r(d.to_node)})
        for d in spec.data_flow
    ]
    for old, new in renames.items():
        print(
            f"  ◌  WCA  node '{old}' reuses catalog '{new}' verbatim — "
            f"renamed node and updated its edge references",
            flush=True,
        )
    return spec.model_copy(update={
        "nodes": final_nodes,
        "edges": edges,
        "data_flow": data_flow,
        "entry": _r(spec.entry),
    })


def _ensure_reachable(spec: WorkflowSpec) -> WorkflowSpec:
    """Guarantee a single entry and that every node is reachable from it.

    FaaSr requires exactly one entry action and rejects any node not reachable
    from it via InvokeNext, which emit.py derives ONLY from invocation edges (not
    data_flow). A data-only producer — its output is consumed downstream but
    nothing triggers it — is therefore a second root and fails deploy validation
    ("unreachable state"). WCA is prompted to avoid this, but we repair it
    deterministically here (before Gate 2) so a slip can't force a full
    deploy-fail → regenerate loop.

    For each unreachable node we add ONE incoming trigger edge, preferring a data
    producer of that node (so it fires only after its input file exists) and
    falling back to the entry (a pure fan-out) for a node with no data producer.
    Edges are only added from already-reachable nodes, and never when the orphan
    can already reach the chosen source, so no cycle can form.
    """
    nodes = spec.nodes
    if not nodes:
        return spec
    names = [n.name for n in nodes]
    name_set = set(names)

    # Drop edges referencing nonexistent nodes (a resolve-LLM slip): a dangling
    # edge would make its real endpoint look orphaned here, then KeyError in the
    # Gate-2 DAG display and in emit at deploy time.
    edges, dropped = [], []
    for e in spec.edges:
        if e.from_node in name_set and e.to_node in name_set:
            edges.append(e)
        else:
            dropped.append(e)
    for e in dropped:
        print(
            f"  ◌  WCA  dropped edge {e.from_node} → {e.to_node}: "
            f"references a node that doesn't exist",
            flush=True,
        )

    # Resolve the entry: declared entry if it names a real node; else a node with
    # no incoming edge; else the first node.
    incoming = {e.to_node for e in edges}
    entry = spec.entry if spec.entry in name_set else None
    if entry is None:
        roots = [n for n in names if n not in incoming]
        entry = roots[0] if roots else names[0]

    # to_node -> data producers (nodes whose file it reads), for a natural trigger.
    producers: dict[str, list[str]] = {}
    for df in spec.data_flow:
        if df.from_node in name_set and df.to_node in name_set and df.from_node != df.to_node:
            producers.setdefault(df.to_node, []).append(df.from_node)

    def _adjacency(edge_list):
        adj: dict[str, list[str]] = {n: [] for n in names}
        for e in edge_list:
            if e.from_node in adj and e.to_node in name_set:
                adj[e.from_node].append(e.to_node)
        return adj

    def _reach(start, adj):
        seen, stack = set(), [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, []))
        return seen

    added: list[tuple[str, str]] = []
    for _ in range(len(names) + 1):
        adj = _adjacency(edges)
        reachable = _reach(entry, adj)
        orphans = [n for n in names if n not in reachable]
        if not orphans:
            break
        progressed = False
        for orphan in orphans:
            prod = [p for p in producers.get(orphan, []) if p in reachable]
            if prod:
                chosen = prod[0]
            elif not producers.get(orphan):
                chosen = entry  # pure source with no data dependency → trigger from entry
            else:
                continue  # has producers but none reachable yet — defer to a later round
            if chosen == orphan or orphan in _reach(chosen, adj):
                continue  # cycle guard
            edges.append(WorkflowEdge(from_node=chosen, to_node=orphan))
            added.append((chosen, orphan))
            progressed = True
        if not progressed:
            break  # can't make further progress (e.g. a data cycle) — let validation report it

    if not added and not dropped:
        return spec
    for frm, to in added:
        print(f"  ◌  WCA  added trigger edge {frm} → {to} to keep it reachable", flush=True)
    return spec.model_copy(update={"edges": edges, "entry": entry})


def _prune_transitive_edges(spec: WorkflowSpec) -> WorkflowSpec:
    """Drop invocation edges another path already implies (transitive reduction).

    The resolve stage gives every data_flow entry a matching edge, so a node that
    reads a file written by a non-adjacent ancestor gets a direct edge from that
    producer ON TOP of the chain that already orders the two — e.g.
    `interpolate → pyadm1` alongside `interpolate → vary_inputs → pyadm1`. That
    extra edge buys nothing: S3 files persist, so the ancestor's output is already
    in the data store by the time the consumer runs, and the surviving path
    enforces the same ordering. What it does cost is a spurious multi-predecessor
    fan-in on the consumer (a second InvokeNext of it, ranked instances included)
    and a branching DAG at Gate 2 where the user described a sequential one.

    An edge u → v is dropped only when v is STILL reachable from u without it, so
    reachability — and therefore `_ensure_reachable`'s single-entry guarantee — is
    preserved by construction. Conditional edges are never dropped and never count
    as an alternate path: a branch that may not fire can't stand in for an
    unconditional invocation.
    """
    remaining = list(spec.edges)
    if len(remaining) < 2:
        return spec

    def _reaches(start: str, target: str, edge_list: list[WorkflowEdge]) -> bool:
        adj: dict[str, list[str]] = {}
        for e in edge_list:
            if e.condition is None:
                adj.setdefault(e.from_node, []).append(e.to_node)
        seen: set[str] = set()
        stack = list(adj.get(start, []))
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, []))
        return False

    removed: list[WorkflowEdge] = []
    for edge in list(remaining):
        if edge.condition is not None:
            continue
        # Identity comparison, so an exact duplicate edge is a valid alternate
        # path for the first copy (dedupe) but not for the last one standing.
        without = [e for e in remaining if e is not edge]
        if _reaches(edge.from_node, edge.to_node, without):
            remaining = without
            removed.append(edge)

    if not removed:
        return spec
    for e in removed:
        print(
            f"  ◌  WCA  dropped redundant edge {e.from_node} → {e.to_node}: "
            f"already ordered by another path",
            flush=True,
        )
    return spec.model_copy(update={"edges": remaining})


def wca_fca_node(state: AgentState) -> dict:
    """
    WCA Phase 2a: FCA catalog lookup + initial resolve.  No interrupt — safe from replay.

    Stores fca_summary and the enriched spec in state so wca_gate2_node can read them
    without re-running FCA on every LangGraph node replay.
    """
    llm = get_llm()
    spec = state["workflow_spec"]

    # ALL standing user directives (accumulated across revision rounds), plus any
    # Gate-3 code_feedback — so the FCA re-decides sourcing with the full history and a
    # prior directive (e.g. "generate viz from new") is never dropped on a later round.
    prior_feedback = _accumulated_feedback(state)
    code_feedback = (state.get("code_feedback") or "").strip()
    if code_feedback and code_feedback not in prior_feedback:
        prior_feedback = (prior_feedback + "\n- " + code_feedback).strip()
    workflow_context = (spec.description or "").strip()
    if spec.name:
        workflow_context = f"{spec.name}: {workflow_context}".strip().rstrip(":").strip()
    # Append the verbatim user request so the FCA grounds sourcing decisions and
    # refined/adapted descriptions in the user's actual words, not just the
    # skeleton's summary of them.
    user_request = (state.get("user_request") or "").strip()
    if user_request:
        workflow_context += f"\nOriginal user request (verbatim): {user_request}"
    # Make the per-node FCA prompt aware of any user-attached reference files (names
    # only — the full bytes stay on disk for the FGA to read).
    context_files = state.get("context_files") or []
    if context_files:
        workflow_context += (
            "\nUser-provided reference files available to the implementer: "
            + ", ".join(c.name for c in context_files)
        )

    # Genuine prior implementations this workflow already has, keyed by node name.
    # ONLY sourced from prior_spec (stashed by the Gate-1 skeleton gate from a previously
    # deployed/implemented workflow). We must NOT read the current in-session
    # workflow_spec here: during a Gate-2 revision loop its coded nodes are unvalidated
    # catalog/adapt SEEDS, not real implementations — treating those as "prior" let the
    # FCA falsely tag a rejected catalog seed as [cached]. Only genuinely-implemented
    # code (from prior_spec) may be offered for reuse as "cached".
    prior_impls: dict[str, FunctionSpec] = {}
    prior_spec = state.get("prior_spec")
    for nd in (getattr(prior_spec, "nodes", None) or []):
        if nd.code:
            prior_impls[nd.name] = nd

    # User-attached scripts/functions, keyed by node name — FCA sources a matching
    # node as [User Provided Model] (see propose_candidate).
    user_models = state.get("user_models") or []
    user_models_by_name = {m.name: m for m in user_models}

    enriched_nodes = []
    n = len(spec.nodes)
    for idx, node in enumerate(spec.nodes, 1):
        print(f"  ◌  FCA  [{idx}/{n}] searching catalog for '{node.name}'...", flush=True)
        candidate = propose_candidate(
            node, feedback=prior_feedback, workflow_context=workflow_context,
            prior_node=prior_impls.get(node.name),
            user_model=user_models_by_name.get(node.name),
        )
        source_tag = {
            "catalog": "[catalog]", "adapt": "[adapt]", "cached": "[cached]",
            "user_provided": "[User Provided Model]",
        }.get(candidate.source, "[new]")
        if candidate.source == "user_provided" and candidate.user_model_mode:
            source_tag = f"[User Provided Model — {candidate.user_model_mode}]"
        origin = (
            f" ← [{candidate.origin_kind}] {candidate.origin_name}"
            if candidate.origin_kind else ""
        )
        # For adapt nodes the description IS the FCA's reasoning (adaptation
        # notes) — print all of it, wrapped, not a truncated teaser.
        desc_lines = textwrap.wrap((candidate.description or "").strip(), width=92) or [""]
        print(f"       → {source_tag}{origin} {desc_lines[0]}", flush=True)
        for line in desc_lines[1:]:
            print(f"         {line}", flush=True)
        enriched_nodes.append(candidate)

    # Rewrite edge/data_flow/entry references for any node the FCA renamed
    # (verbatim catalog reuse) BEFORE the resolve LLM and reachability repair
    # see the spec, so they never see dangling skeleton names.
    spec_with_candidates = _apply_candidate_renames(spec, enriched_nodes)

    fca_summary: list[dict] = [
        {
            "name":        candidate.name,
            "source":      candidate.source,
            "user_model_mode": candidate.user_model_mode,
            "catalog_id":  candidate.catalog_id,
            "origin_kind": candidate.origin_kind,
            "origin_name": candidate.origin_name,
            "description": candidate.description,
            "inputs":      [i.name for i in candidate.inputs],
            "outputs":     [o.name for o in candidate.outputs],
            "dependencies": candidate.dependencies or [],
            "secrets":     candidate.secrets or [],
        }
        for candidate in spec_with_candidates.nodes
    ]

    print(f"  ◌  WCA  resolving data-flow edges...", flush=True)

    feedback_section = ""
    if prior_feedback.strip():
        feedback_section = (
            f"\nStanding change requests (honor ALL — later items override earlier ones "
            f"on conflict):\n{prior_feedback}\n\n"
            "Apply every one of these to the workflow.\n"
        )
    prompt = WCA_RESOLVE_PROMPT.format(
        workflow_json=spec_with_candidates.model_dump_json(indent=2),
        fca_candidates=_fca_candidates_text(fca_summary),
        feedback_section=feedback_section,
        user_models_section=_user_models_section(user_models),
        context_files_section=_context_files_section(context_files),
    )
    messages = [SystemMessage(content=WCA_SYSTEM), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    record_usage(response, "WCA")
    raw = response.content

    try:
        resolved_spec = _merge_resolved(raw, spec_with_candidates.nodes)
    except Exception:
        resolved_spec = spec_with_candidates

    # Enforce single-entry + reachability so a data-only producer can't slip
    # through as an orphaned second root and fail FaaSr deploy validation, then
    # drop any edge the surviving paths already imply (repair first, so pruning
    # can never race a just-added trigger edge).
    resolved_spec = _prune_transitive_edges(_ensure_reachable(resolved_spec))

    return {
        "workflow_spec": resolved_spec,
        "wca_fca_summary": fca_summary,
        "wca_enriched_spec": spec_with_candidates,
        "messages": [AIMessage(content=raw)],
    }


def wca_gate2_node(state: AgentState) -> dict:
    """
    WCA Phase 2b: Gate 2 — display the resolved workflow and capture one decision.

    Pure display + decision gate (single interrupt, replay-safe):
      - "approve" → proceed to FGA; clear feedback and the cached FCA state.
      - a FUNCTION change → store it as feedback and loop back to wca_fca, which
        re-runs the FCA (so a sourcing change like "adapt X, don't reuse it" is
        honored) and re-resolves data flow before returning here.
      - a STRUCTURE change (rewire/add/remove/reorder steps, change a rank) →
        route back to the skeleton stage, where edges are actually decided. The
        resolve stage can only rewrite I/O around the topology it is handed, so a
        "make these steps sequential" request asked of it tends to come back
        unchanged; Gate 3 already classifies and routes this way.
    """
    fca_summary: list[dict] = state.get("wca_fca_summary") or []
    resolved_spec = state["workflow_spec"]
    iteration = state.get("iteration", 0)
    prior_feedback = state.get("feedback") or ""

    hitl_response = interrupt({
        "gate": 2,
        "message": (
            "Review the fleshed-out workflow with function candidates and data flow. "
            "Type 'approve' to proceed to implementation, or describe changes."
        ),
        "fca_summary": fca_summary,
        "workflow_spec": resolved_spec.model_dump() if resolved_spec else None,
        "iteration": iteration,
        "prior_feedback": prior_feedback,
        "external_inputs": external_inputs(resolved_spec) if resolved_spec else [],
    })

    if isinstance(hitl_response, str) and hitl_response.strip().lower() == "approve":
        return {
            "workflow_spec": resolved_spec,
            "messages": [AIMessage(content="WCA Gate 2: approved")],
            "hitl_gate": 2,
            "hitl_decision": "approve",
            "wca_fca_summary": None,
            "wca_enriched_spec": None,
            "feedback": None,
            "structural_change": None,
            # Approval releases the spec to the FGA, which replaces the seeds with
            # real implementations — so a later Gate-3 structural change must be
            # free to stash it as prior_spec again.
            "spec_is_seed": None,
        }

    # Revision: a topology change goes back to the skeleton stage; anything else
    # loops through wca_fca so the FCA re-decides sourcing and the resolve LLM
    # re-applies the change.
    refinement = hitl_response if isinstance(hitl_response, str) else str(hitl_response)
    structural = classify_structural_change(refinement, resolved_spec)
    return {
        "feedback": refinement,
        "revision_directives": [refinement],  # accumulate so prior directives persist
        "structural_change": structural,
        # The Gate-2 spec's code is unvalidated catalog/adapt SEEDS, not real
        # implementations. Tell the skeleton gate not to stash it as prior_spec —
        # the FCA would otherwise offer those seeds back as [cached] "prior work".
        "spec_is_seed": True,
        "messages": [AIMessage(content=(
            "WCA Gate 2: structural change requested" if structural
            else "WCA Gate 2: changes requested"
        ))],
        "hitl_gate": 2,
        "hitl_decision": "revise",
    }
