from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from faasr_agents.state import AgentState
from faasr_agents.agents.wca import (
    wca_compose_skeleton_node,
    wca_skeleton_gate_node,
    wca_fca_node,
    wca_gate2_node,
)
from faasr_agents.agents.fga import fga_node
from faasr_agents.agents.wda import wda_gate4_node, wda_deploy_node
from langchain_core.messages import AIMessage


def code_review_node(state: AgentState) -> dict:
    """
    Gate 3 — let the user browse the generated Python before approving.

    Interrupts with a ``review`` payload consumed by cli.py's _display_interrupt
    (which launches the interactive Rich viewer).  On resume:
      - "approve" / "yes" / "y" → clear code_feedback, proceed to wda
      - anything else           → store as code_feedback, loop back to fga
    """
    from langgraph.types import interrupt

    spec = state["workflow_spec"]

    review = [
        {
            "name": n.name,
            "source": n.source,
            "user_model_mode": n.user_model_mode,
            "code": n.code or "",
            "tests": n.tests or "",
        }
        for n in spec.nodes
    ]

    def _fmt(n):
        return f"  - {n.name} ({n.source}, {len((n.code or '').splitlines())} lines)"

    implemented = [n for n in spec.nodes if n.source not in ("catalog", "cached")]
    reused = [n for n in spec.nodes if n.source in ("catalog", "cached")]

    sections = []
    if implemented:
        sections.append(
            f"FGA implemented {len(implemented)} function(s):\n"
            + "\n".join(_fmt(n) for n in implemented)
        )
    if reused:
        sections.append(
            f"Reused {len(reused)} function(s) (catalog/cached):\n"
            + "\n".join(_fmt(n) for n in reused)
        )

    from faasr_agents.faasr.emit import external_inputs, emit_faasr_json

    response = interrupt({
        "gate": 3,
        "review": review,
        "workflow_name": spec.name,
        "external_inputs": external_inputs(spec),
        "workflow_json": emit_faasr_json(spec),
        "message": (
            "\n\n".join(sections)
            + "\n\nBrowse the generated code in the viewer, then approve or describe changes."
        ),
    })

    if isinstance(response, str) and response.strip().lower() in ("approve", "yes", "y"):
        # Reset structural_change so a stale flag can't misroute the approve path.
        return {
            "code_feedback": None,
            "structural_change": None,
            "messages": [AIMessage(content="Code review: approved")],
        }

    # Explicit change request: classify STRUCTURE (topology) vs FUNCTION (code/IO of
    # existing steps) and route accordingly.
    from faasr_agents.agents.wca import classify_structural_change

    if classify_structural_change(response, spec):
        # Structural → skeleton stage. Put the request on `feedback` (what the
        # skeleton node reads) and clear code_feedback.
        return {
            "feedback": response,
            "code_feedback": None,
            "structural_change": True,
            "revision_directives": [response],  # accumulate so prior directives persist
            "messages": [AIMessage(content="Code review: structural change requested")],
        }
    return {
        "code_feedback": response,
        "structural_change": False,
        "revision_directives": [response],  # accumulate so prior directives persist
        # Function-only change routes straight to fca_resolve, skipping the skeleton
        # (the only other place prior_spec is stashed). Stash the current code-bearing
        # spec here so the FCA can reuse this workflow's own implementations verbatim
        # ([cached]) / as adapt seeds. At Gate 3 the nodes carry real FGA code (not
        # Gate-2 seeds), so this respects the safety invariant in wca_fca_node.
        "prior_spec": spec,
        "messages": [AIMessage(content="Code review: changes requested")],
    }


def branch_after_gate1(state: AgentState) -> Literal["fca_resolve", "compose_skeleton"]:
    # Approval releases the skeleton to resolution; a refinement re-enters the
    # compose node, which continues the Gate-1 conversation from skeleton_draft.
    return "fca_resolve" if state.get("hitl_decision") == "approve" else "compose_skeleton"


def branch_after_code_review(state: AgentState) -> Literal["compose_skeleton", "fca_resolve", "gate4_deploy"]:
    # A structural change re-composes from the skeleton so topology edits take effect;
    # a function-only change routes through FCA so it can re-decide sourcing before FGA re-runs.
    if state.get("structural_change"):
        return "compose_skeleton"
    return "fca_resolve" if state.get("code_feedback") else "gate4_deploy"


def branch_after_gate2(state: AgentState) -> Literal["fca_resolve", "fga_generate", "compose_skeleton"]:
    # Approval proceeds to FGA. A revision loops back to the FCA so it can re-decide
    # sourcing (e.g. flip catalog → adapt) with the feedback — unless it is a
    # structural change, which re-composes from the skeleton so topology edits
    # (rewiring edges, changing a rank) are made where edges are decided.
    if state.get("hitl_decision") != "revise":
        return "fga_generate"
    return "compose_skeleton" if state.get("structural_change") else "fca_resolve"


def branch_after_gate4(state: AgentState) -> Literal["deploy_execute", "compose_skeleton", "fca_resolve", "gate5_output_review", "__end__"]:
    # Approval proceeds to the deploy node; every other Gate-4 outcome
    # (validation failure, skip, decline, revision) routes exactly as before.
    if state.get("hitl_gate") == 4 and state.get("hitl_decision") == "approve":
        return "deploy_execute"
    return branch_after_deploy(state)


def branch_after_deploy(state: AgentState) -> Literal["compose_skeleton", "fca_resolve", "gate5_output_review", "__end__"]:
    """After WDA: output review on success AND execution failure; validation loops back."""
    result = state.get("deploy_result") or {}
    if result.get("success"):
        return "gate5_output_review"
    # An explicit user change request at Gate 4: a structural change re-composes from
    # the skeleton so topology edits (e.g. rank/fan-out N) take effect; a function-only
    # change routes to resolution (fca_resolve), avoiding a needless skeleton round-trip.
    if state.get("hitl_gate") == 4 and state.get("hitl_decision") == "revise":
        return "compose_skeleton" if state.get("structural_change") else "fca_resolve"
    # An execution failure goes through the post-run review (Gate 5): the WDA
    # summary diagnoses the run and may RECOMMEND a revision; the user decides.
    if result.get("failure_class") == "execution":
        return "gate5_output_review"
    iteration = state.get("iteration", 0)
    if iteration >= 5:
        return "__end__"
    if state.get("feedback"):
        return "fca_resolve"
    return "__end__"


def branch_after_output(state: AgentState) -> Literal["fca_resolve", "gate5_output_review", "__end__"]:
    """After output review: change request → WCA; save → reopen the REPL; else done."""
    if state.get("feedback"):
        return "fca_resolve"
    if state.get("review_continue"):
        return "gate5_output_review"
    return "__end__"



def _do_save(spec, save_option: str) -> str:
    """Run catalog/registry saves per save_option. Returns a summary."""
    from faasr_agents.catalog.store import CatalogStore
    from faasr_agents.catalog.workflow_store import WorkflowRegistry
    from faasr_agents.models import CatalogEntry, WorkflowEntry
    import uuid

    parts = []

    if save_option in ("functions", "both"):
        store = CatalogStore()
        stored = []
        skipped = []
        existing_codes = {
            e.function_spec.code.strip()
            for e in store.list_all()
            if e.function_spec.code
        }
        for node in spec.nodes:
            if not node.code:
                continue
            # Only new/adapted implementations enter the catalog. A verbatim
            # catalog reuse is already there (node.catalog_id), and a cached
            # node that originated from the catalog keeps byte-identical code
            # across revision loops even though its source becomes "cached" —
            # the code check catches it (and re-saves of the same workflow).
            if node.source == "catalog" or node.code.strip() in existing_codes:
                skipped.append(node.name)
                continue
            entry = CatalogEntry(
                id=str(uuid.uuid4()),
                function_spec=node,
                keywords=node.name.split("_") + node.description.lower().split()[:5],
                provenance=f"workflow:{spec.name}",
            )
            store.add(entry)
            stored.append(node.name)
            existing_codes.add(node.code.strip())
        parts.append(f"Stored {len(stored)} function(s) in catalog: {stored}.")
        if skipped:
            parts.append(f"Skipped {len(skipped)} already-cataloged: {skipped}.")
    if save_option in ("workflow", "both"):
        registry = WorkflowRegistry()
        existing = registry.get_by_name(spec.name)
        wf_entry = WorkflowEntry(
            id=existing.id if existing else str(uuid.uuid4()),
            workflow_spec=spec,
            keywords=spec.name.replace("-", " ").split() + spec.description.lower().split()[:5],
            provenance="user request",
            usage_count=existing.usage_count if existing else 0,
        )
        registry.add(wf_entry)
        verb = "Updated" if existing else "Saved"
        parts.append(f"{verb} workflow '{spec.name}' for reuse.")
    return " ".join(parts) if parts else "Nothing saved."


def output_review_node(state: AgentState) -> dict:
    """
    HITL Gate 5 — post-run output review + optional save for reuse.

    Interrupts with the run context so cli.py can launch the interactive REPL.
    On resume:
      "accept"          → no saving, proceed to END
      "save:*"          → run catalog/registry save, proceed to END
      <feedback text>   → store as feedback, loop back to fca_resolve
    """
    from langgraph.types import interrupt
    from langchain_core.messages import AIMessage
    from faasr_agents.faasr.emit import emit_faasr_json

    spec = state["workflow_spec"]
    deploy_result = state.get("deploy_result") or {}

    workflow_json = emit_faasr_json(spec)

    # Derive output artifacts from the emitted ActionList Arguments
    from faasr_agents.faasr.artifacts import build_artifact_list
    artifacts = build_artifact_list(workflow_json)

    code_by_node = {node.name: node.code or "" for node in spec.nodes}
    logs = deploy_result.get("logs", "")
    success = deploy_result.get("success", True)

    if success:
        message = (
            f"Workflow '{spec.name}' executed successfully. "
            "Review the outputs below, then accept, save, or request changes."
        )
    else:
        message = (
            f"Workflow '{spec.name}' FAILED during execution. "
            "Review the logs and outputs below, then accept the failure or request a revision."
        )

    response = interrupt({
        "gate": 5,
        # False when re-entering after a `save` loop-back — lets the REPL skip
        # regenerating the (expensive) WDA run summary it already showed.
        "first_visit": not bool(state.get("review_continue")),
        "workflow_json": workflow_json,
        "artifacts": artifacts,
        "logs": logs,
        "code_by_node": code_by_node,
        "workflow_name": spec.name,
        "success": success,
        "failed_functions": deploy_result.get("failed_functions", []),
        "message": message,
    })

    if isinstance(response, str):
        r = response.strip()
        if r == "accept":
            return {
                "feedback": None,
                "review_continue": False,
                "messages": [AIMessage(content="Output review: accepted, no saving")],
            }
        if r.startswith("save:"):
            save_option = r[5:]  # "functions" | "workflow" | "both"
            summary = _do_save(spec, save_option)
            print(f"  {summary}")
            # A save is not a terminal action: loop back into the review REPL so
            # the user can keep working (save the rest, ask questions, request
            # changes) and exit deliberately with 'accept'.
            return {
                "feedback": None,
                "review_continue": True,
                "messages": [AIMessage(content=f"Output review: saved ({save_option}). {summary}")],
            }
        # Anything else is a change request
        return {
            "feedback": r,
            "revision_directives": [r],  # accumulate so prior directives persist
            "review_continue": False,
            "iteration": state.get("iteration", 0) + 1,
            # Gate-5 change routes to fca_resolve, skipping the skeleton. Stash the
            # just-run spec (genuinely-implemented, deployed code) so the FCA can reuse
            # untouched functions verbatim ([cached]) and adapt the targeted one from
            # its own prior code rather than re-searching the catalog.
            "prior_spec": spec,
            "messages": [AIMessage(content="Output review: changes requested")],
        }

    return {
        "feedback": None,
        "review_continue": False,
        "messages": [AIMessage(content="Output review: complete")],
    }


def build_graph(entry_node: str = "compose_skeleton") -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("compose_skeleton", wca_compose_skeleton_node)
    graph.add_node("gate1_skeleton", wca_skeleton_gate_node)
    graph.add_node("fca_resolve", wca_fca_node)
    graph.add_node("gate2_candidates", wca_gate2_node)
    graph.add_node("fga_generate", fga_node)
    graph.add_node("gate3_code_review", code_review_node)
    graph.add_node("gate4_deploy", wda_gate4_node)
    graph.add_node("deploy_execute", wda_deploy_node)
    graph.add_node("gate5_output_review", output_review_node)

    graph.set_entry_point(entry_node)
    graph.add_edge("compose_skeleton", "gate1_skeleton")
    graph.add_conditional_edges(
        "gate1_skeleton",
        branch_after_gate1,
        {"fca_resolve": "fca_resolve", "compose_skeleton": "compose_skeleton"},
    )
    graph.add_edge("fca_resolve", "gate2_candidates")
    graph.add_conditional_edges(
        "gate2_candidates",
        branch_after_gate2,
        {
            "fca_resolve": "fca_resolve",
            "fga_generate": "fga_generate",
            "compose_skeleton": "compose_skeleton",
        },
    )
    graph.add_edge("fga_generate", "gate3_code_review")
    graph.add_conditional_edges(
        "gate3_code_review",
        branch_after_code_review,
        {"compose_skeleton": "compose_skeleton", "fca_resolve": "fca_resolve", "gate4_deploy": "gate4_deploy"},
    )
    graph.add_conditional_edges(
        "gate4_deploy",
        branch_after_gate4,
        {
            "deploy_execute": "deploy_execute",
            "compose_skeleton": "compose_skeleton",
            "fca_resolve": "fca_resolve",
            "gate5_output_review": "gate5_output_review",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "deploy_execute",
        branch_after_deploy,
        {
            "compose_skeleton": "compose_skeleton",
            "fca_resolve": "fca_resolve",
            "gate5_output_review": "gate5_output_review",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "gate5_output_review",
        branch_after_output,
        {"fca_resolve": "fca_resolve", "gate5_output_review": "gate5_output_review", "__end__": END},
    )

    return graph


def compile_graph(checkpointer=None, entry_node: str = "compose_skeleton"):
    graph = build_graph(entry_node=entry_node)
    cp = checkpointer or MemorySaver()
    return graph.compile(
        checkpointer=cp,
        interrupt_before=[],  # interrupts are managed via interrupt() calls inside nodes
    )
