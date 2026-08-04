from __future__ import annotations
import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from faasr_agents.models import WorkflowSpec, UserModel, ContextFile


class AgentState(TypedDict):
    user_request: str
    workflow_spec: Optional[WorkflowSpec]
    messages: Annotated[list, add_messages]
    # HITL
    hitl_decision: Optional[str]          # "approve" | "refine:<instructions>"
    # WDA result
    deploy_result: Optional[dict]         # {"success": bool, "logs": str, "errors": str}
    # feedback from WDA back to WCA
    feedback: Optional[str]
    # Accumulated EXPLICIT user change-request directives across all gates (append-only
    # for the session). Overwriting `feedback` each revision loses prior directives, so
    # the WCA/FCA/resolve prompts render this full history every round — a directive like
    # "generate viz from new" persists and isn't undone by a later unrelated request.
    revision_directives: Annotated[list[str], operator.add]
    iteration: int
    # which HITL gate we're at (1-5)
    hitl_gate: Optional[int]
    # whole-batch code-change feedback from Gate 3 back to FGA
    code_feedback: Optional[str]
    # Routing hint set by Gate 2 / Gate 3 / Gate 4 on an EXPLICIT user change request: True if
    # the change alters workflow structure (topology) → route to the skeleton stage;
    # False if it only changes functions → route to the resolution stage. Set fresh
    # per change request (and reset on Gate-3 approve); read by the branch functions.
    structural_change: Optional[bool]
    # FCA resolve cache: set by wca_fca_node so wca_gate2_node doesn't re-run FCA on replay
    wca_fca_summary: Optional[list]
    wca_enriched_spec: Optional[WorkflowSpec]
    # Gate 5: after a save, reopen the output-review REPL instead of ending the run
    review_continue: Optional[bool]
    # Most recent code-bearing spec, stashed by the Gate-1 skeleton gate before it
    # strips code from workflow_spec — lets the FCA reuse this workflow's own prior
    # implementations (source="cached") across a topology-changing revision.
    prior_spec: Optional[WorkflowSpec]
    # True while workflow_spec's code is only Gate-2 catalog/adapt SEEDS (nothing the
    # FGA has generated or validated). Set when a Gate-2 structural revision routes
    # back to the skeleton; read by the Gate-1 gate so it doesn't stash those seeds as
    # prior_spec and let the FCA re-offer them as [cached]. Cleared on Gate-1 approval.
    spec_is_seed: Optional[bool]
    # In-flight Gate-1 skeleton draft, shared between the compose node and the gate:
    # {"conversation": [messages], "raw": str, "spec": WorkflowSpec|None,
    #  "parse_error": str|None, "last_good": WorkflowSpec|None, "refinement": str|None}.
    # A pending "refinement" tells the compose node to continue the conversation
    # instead of starting fresh; cleared (None) on Gate-1 approval.
    skeleton_draft: Optional[dict]
    # Artifacts dir minted by _save_artifacts at Gate 4 — each call creates a unique
    # dir, so the deploy node must reference the gate's dir rather than re-derive it.
    artifacts_dir: Optional[str]
    # User-attached scripts/functions to bring into this run as [User Provided Model]
    # nodes. Set once at init from the startup picker; read-only thereafter.
    user_models: Optional[list[UserModel]]
    # User-attached non-code reference files (papers, I/O examples, datasets). Copied
    # into the FGA context/ folder and listed for the planners; NOT workflow nodes.
    # Set once at init from the startup picker; read-only thereafter.
    context_files: Optional[list[ContextFile]]
