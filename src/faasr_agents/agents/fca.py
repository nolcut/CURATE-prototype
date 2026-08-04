from __future__ import annotations
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from faasr_agents.llm import get_llm
from faasr_agents.pricing import record_usage
from faasr_agents.catalog.store import CatalogStore
from faasr_agents.models import FunctionSpec, IOSpec, clean_dependencies
from faasr_agents.prompts.fca_prompts import (
    FCA_SYSTEM,
    FCA_CANDIDATE_PROMPT,
    FCA_USER_MODEL_PROMPT,
)

_catalog = CatalogStore()


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _coerce_iospecs(raw) -> list[IOSpec] | None:
    """Coerce LLM-provided I/O (list of dicts or strings) into IOSpecs.

    Returns None when there is nothing usable, so callers can fall back to the
    catalog function's original I/O.
    """
    if not isinstance(raw, list):
        return None
    specs: list[IOSpec] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            specs.append(IOSpec(
                name=item["name"],
                type=item.get("type", "any"),
                description=item.get("description", ""),
            ))
        elif isinstance(item, str) and item.strip():
            specs.append(IOSpec(name=item.strip()))
    return specs or None


def _uniq(*sources) -> list[str]:
    """Strip + dedupe secret names across lists, preserving order. No coercion —
    the name is used verbatim in the code, workflow.json, and the user list."""
    seen: list[str] = []
    for src in sources:
        for raw in src or []:
            s = str(raw).strip()
            if s and s not in seen:
                seen.append(s)
    return seen


def _decide_user_model_mode(spec: FunctionSpec, user_model, feedback: str,
                            workflow_context: str) -> str:
    """Verbatim-vs-adapt decision for a user-provided model.

    Only the model's OWN computation needing to change justifies "adapt" —
    FaaSr-compatibility issues are absorbed by the wrapper. Biased to
    "verbatim": any parse failure or LLM error falls back to it, and an
    explicit user directive to use the model verbatim is a hard override
    (enforced in the prompt; the directive persists in the accumulated
    feedback across revision rounds, so the override sticks).
    """
    try:
        llm = get_llm()
        feedback_section = (
            f"User request revisions / feedback:\n{feedback}" if feedback.strip()
            else "(no revisions or feedback)"
        )
        prompt = FCA_USER_MODEL_PROMPT.format(
            function_name=spec.name,
            function_description=spec.description or "(none)",
            inputs=", ".join(i.name for i in spec.inputs) or "(unresolved)",
            outputs=", ".join(o.name for o in spec.outputs) or "(unresolved)",
            workflow_context=workflow_context or "(none)",
            feedback_section=feedback_section,
            model_code=user_model.code[:3000],
        )
        response = llm.invoke([SystemMessage(content=FCA_SYSTEM), HumanMessage(content=prompt)])
        record_usage(response, "FCA")
        result = _extract_json(response.content)
        mode = str(result.get("mode", "")).strip().lower()
    except Exception:
        mode = ""
    return "adapt" if mode == "adapt" else "verbatim"


def propose_candidate(
    spec: FunctionSpec,
    feedback: str = "",
    workflow_context: str = "",
    prior_node: FunctionSpec | None = None,
    user_model=None,
) -> FunctionSpec:
    """
    Given a FunctionSpec skeleton (name, description, inputs, outputs),
    search the catalog and return an enriched FunctionSpec with:
    - source="user_provided" if the user attached a script/function for this node
      (matched by name) — seed its original code; FGA refactors + verifies it
    - source="cached" if reusing this workflow's OWN prior implementation verbatim
    - source="catalog" and catalog_id set if reusing a catalog function verbatim
    - source="adapt" and catalog_id set if reusing a catalog function as a seed to regenerate
    - source="new" if the function must be implemented from scratch

    Pass feedback to let the FCA LLM account for change requests when deciding
    whether to reuse, adapt, or regenerate from scratch.  Pass workflow_context
    (the overall workflow intent) so the FCA can detect a catalog match whose
    domain specifics conflict with the workflow and prefer adapt over reuse.
    Pass prior_node (this workflow's existing implementation of the same function,
    when revising) so the FCA can choose "cached" and reuse it verbatim instead of
    regenerating an unchanged function.

    Pass user_model (a UserModel the user attached for this exact node name) to source
    it as "user_provided": its original code becomes the FGA seed, and I/O is resolved
    by the WCA like an "adapt" node.
    """
    # A user-attached script/function for this node wins outright — it's a guaranteed
    # step, so short-circuit the catalog decision and seed the original code. The only
    # question left is the mode: wrap the model verbatim, or adapt its computation.
    if user_model is not None and getattr(user_model, "code", None):
        return spec.model_copy(update={
            "source": "user_provided",
            "user_model_mode": _decide_user_model_mode(spec, user_model, feedback, workflow_context),
            "catalog_id": None,
            "origin_kind": None,
            "origin_name": None,
            "code": user_model.code,
            "description": (user_model.description or spec.description or "").strip(),
        })

    llm = get_llm()

    # Tokenize the node name the same way the catalog corpus tokenizes entry
    # names (store.py splits spec.name on "_"); otherwise "map_transcription"
    # enters the query as one unmatchable token and the node-name signal is lost.
    name_tokens = spec.name.replace("_", " ")
    query = f"{name_tokens} {spec.description} {' '.join(i.name for i in spec.inputs)} {' '.join(o.name for o in spec.outputs)}"
    matches = _catalog.search(query, k=5)

    catalog_text = "No catalog matches found."
    if matches:
        catalog_text = "\n".join(
            f"[{e.id}] {e.function_spec.name}: {e.function_spec.description} "
            f"| inputs={[i.name for i in e.function_spec.inputs]} "
            f"| outputs={[o.name for o in e.function_spec.outputs]}"
            for e in matches
        )

    feedback_section = ""
    if feedback.strip():
        feedback_section = f"\nChange request / feedback:\n{feedback.strip()}\n"

    has_prior = prior_node is not None and bool(prior_node.code)
    prior_impl_section = ""
    if has_prior:
        prior_impl_section = (
            "\nThis function ALREADY HAS a working implementation in the workflow "
            "being revised:\n"
            f"  description: {prior_node.description}\n"
            f"  inputs={[i.name for i in prior_node.inputs]} "
            f"outputs={[o.name for o in prior_node.outputs]}\n"
            "If the revision does NOT change what this function does, choose "
            "decision \"cached\" to reuse this existing implementation verbatim "
            "(do not regenerate it). If the revision DOES change this function, "
            "choose decision \"adapt\" with adapt_from \"cached\" so it is "
            "regenerated starting from THIS existing implementation as the seed "
            "(preferred over a catalog seed or a from-scratch rewrite).\n"
        )

    prompt = FCA_CANDIDATE_PROMPT.format(
        function_name=spec.name,
        function_description=spec.description,
        inputs=[{"name": i.name, "type": i.type} for i in spec.inputs],
        outputs=[{"name": o.name, "type": o.type} for o in spec.outputs],
        catalog_matches=catalog_text,
        feedback_section=feedback_section,
        workflow_context=workflow_context.strip() or "(not specified)",
        prior_impl_section=prior_impl_section,
    )

    messages = [SystemMessage(content=FCA_SYSTEM), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    record_usage(response, "FCA")  # registers into the run cost accumulator

    try:
        result = _extract_json(response.content)
    except Exception:
        return spec.model_copy(update={"source": "new"})

    decision = result.get("decision", "new")
    adapt_from = result.get("adapt_from")

    # Reuse this workflow's own prior implementation verbatim. rank comes from the
    # new skeleton in case the revision changed fan-out; everything else is kept.
    if decision == "cached" and has_prior:
        return prior_node.model_copy(update={
            "source": "cached", "rank": spec.rank,
            "origin_kind": "cache", "origin_name": prior_node.name,
        })

    # Adapt seeded from this workflow's OWN prior implementation. Chosen when the FCA
    # says adapt_from="cached", or when it asked to adapt but named no usable catalog
    # item — in both cases the prior code is the seed (better than an unrelated catalog
    # seed or a from-scratch rewrite). Keep prior_node.code as the FGA seed; take the
    # revised description/I/O so the regeneration can change them. rank from the skeleton.
    if decision == "adapt" and has_prior and (
        adapt_from == "cached" or not result.get("catalog_id")
    ):
        updates = {
            "source": "adapt",
            "catalog_id": None,
            "origin_kind": "cache",
            "origin_name": prior_node.name,
            "rank": spec.rank,
            "description": result.get("adaptation_notes")
                or result.get("refined_description")
                or prior_node.description,
        }
        new_inputs = _coerce_iospecs(result.get("updated_inputs"))
        new_outputs = _coerce_iospecs(result.get("updated_outputs"))
        if new_inputs is not None:
            updates["inputs"] = new_inputs
        if new_outputs is not None:
            updates["outputs"] = new_outputs
        return prior_node.model_copy(update=updates)

    if decision in ("reuse", "adapt"):
        catalog_id = result.get("catalog_id")
        if catalog_id:
            entry = _catalog.get(catalog_id)
            if entry:
                _catalog.increment_usage(catalog_id)
                source = "adapt" if decision == "adapt" else "catalog"
                updates = {
                    "source": source,
                    "catalog_id": catalog_id,
                    "origin_kind": "catalog",
                    "origin_name": entry.function_spec.name,
                    "description": result.get("adaptation_notes") or entry.function_spec.description,
                    # union of the catalog function's own secrets + any newly declared
                    "secrets": _uniq(
                        entry.function_spec.secrets, result.get("required_secrets")
                    ),
                    "dependencies": clean_dependencies(entry.function_spec.dependencies),
                    "rank": spec.rank,  # rank is topology (from skeleton), not the catalog item
                }
                # Only adapt nodes get renamed I/O; catalog reuse stays byte-identical.
                if source == "adapt":
                    # Adapt is regenerated by FGA (def <node.name>), so carry the
                    # meaning-bearing skeleton name rather than the stale catalog name.
                    updates["name"] = spec.name
                    new_inputs = _coerce_iospecs(result.get("updated_inputs"))
                    new_outputs = _coerce_iospecs(result.get("updated_outputs"))
                    if new_inputs is not None:
                        updates["inputs"] = new_inputs
                    if new_outputs is not None:
                        updates["outputs"] = new_outputs
                return entry.function_spec.model_copy(update=updates)

    return spec.model_copy(update={
        "source": "new",
        "catalog_id": None,
        # Clear any stale provenance carried over from a prior round's adapt/catalog
        # decision — a "new" node must not show a "← [catalog] …" origin.
        "origin_kind": None,
        "origin_name": None,
        "description": result.get("refined_description") or spec.description,
        "dependencies": clean_dependencies(result.get("suggested_dependencies") or spec.dependencies),
        "secrets": _uniq(spec.secrets, result.get("required_secrets")),
    })
