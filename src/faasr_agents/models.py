from __future__ import annotations
import sys
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


def clean_dependencies(deps) -> list[str]:
    """Keep only real third-party PyPI packages.

    Drops the FaaSr runtime (faasr / FaaSr_py — injected, never pip-installed) and
    any standard-library module (re, json, os, collections, …) which isn't on PyPI.
    Dedupes, preserving order.
    """
    _runtime = {"faasr", "faasr_py", "faasrpy"}
    _stdlib = set(sys.stdlib_module_names)
    out: list[str] = []
    for raw in deps or []:
        name = str(raw).strip()
        if not name:
            continue
        base = name.lower().replace("-", "_")
        if base in _runtime or base in _stdlib:
            continue
        if name not in out:
            out.append(name)
    return out


class IOSpec(BaseModel):
    name: str
    type: str = "any"
    description: str = ""

    @field_validator("name")
    @classmethod
    def _flatten_name(cls, v: str) -> str:
        """Filenames on the data store are flat — the S3 folder is a separate
        FaaSr argument"""
        return v.rsplit("/", 1)[-1].strip()


class FunctionSpec(BaseModel):
    name: str
    description: str = ""
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # PyPI package names
    secrets: list[str] = Field(default_factory=list)  # uppercase env-var names, read via faasr_secret()
    rank: int = 1  # >1 → ranked/parallel: invoked as Name(N), each instance gets faasr_rank()
    # "user_provided": seeded from a user-attached script/function; the FCA decides
    # user_model_mode (verbatim wrap vs adapt), the FGA executes it and verifies parity.
    source: Literal["catalog", "adapt", "new", "cached", "user_provided"] = "new"
    # Only for source == "user_provided". "verbatim": the model's code must not change —
    # the FGA copies it unchanged and wraps it with FaaSr I/O + a compliant entry function
    # that CALLS the model. "adapt": the model's own computation must change per the
    # user's request; the FGA refactors it. FaaSr-compatibility issues alone never
    # justify "adapt" — the wrapper absorbs them.
    user_model_mode: Optional[Literal["verbatim", "adapt"]] = None
    catalog_id: Optional[str] = None
    code: Optional[str] = None  # Python function source
    tests: Optional[str] = None  # Python test source
    # Provenance for reused/adapted/cached nodes: where the code came from.
    # "catalog" → seeded/reused from a catalog function; "cache" → this workflow's
    # own genuinely-implemented prior code. origin_name is the source function name.
    origin_kind: Optional[Literal["catalog", "cache"]] = None
    origin_name: Optional[str] = None


class UserModel(BaseModel):
    """A user-attached Python artifact (a script OR a function) to bring into a run.

    `code` is the full original source, verbatim — do not assume it defines a single
    function; it may be a plain script (top-level statements, argparse, __main__). The
    FGA refactors it into a FaaSr-compatible function and verifies behavior parity.
    Attached as a "[User Provided Model]" node (FunctionSpec.source == "user_provided").
    """
    name: str          # snake_case node name (derived from the filename stem)
    code: str          # full original source
    description: str = ""  # module docstring, if any
    path: str = ""     # original file path (provenance)


class ContextFile(BaseModel):
    """A user-attached non-code reference file (paper, I/O example, dataset, doc).

    Unlike a UserModel, this is NOT a workflow node. Its bytes are copied verbatim
    into the FGA scratch `context/` folder so the implementation agent can read them
    for domain grounding, and its filename is surfaced to the planning agents (WCA/FCA).
    """
    name: str          # basename, e.g. "adm1_paper.pdf"
    path: str          # absolute source path (copied into context/ at seed time)
    description: str = ""  # optional short note / derived hint


class DataFlowEdge(BaseModel):
    from_node: str
    to_node: str
    file: str  # concrete filename on the data store
    format: str = ""  # e.g. "csv", "json", "png"
    folder: str = "data"


class WorkflowEdge(BaseModel):
    from_node: str
    to_node: str
    condition: Optional[Literal["True", "False"]] = None  # for conditional branching


class WorkflowSpec(BaseModel):
    name: str
    description: str = ""
    nodes: list[FunctionSpec] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    data_flow: list[DataFlowEdge] = Field(default_factory=list)
    entry: str = ""  # name of the entry-point function
    compute_target: str = "GH"  # key into ComputeServers
    data_store: str = "S3"  # key into DataStores
    folder: str = "workflow_data"  # S3 folder prefix


# Run-state / provenance fields on FunctionSpec that describe how a *particular run*
# sourced or created a function. They are always re-derived when a function is reused
# (the FCA re-stamps them per node; the CLI re-stamps them for whole-workflow reuse),
# so their stored values are never read. They are excluded when persisting catalog and
# workflow entries to keep the on-disk catalog free of stale run-state.
CATALOG_EXCLUDED_SPEC_FIELDS = {"tests", "origin_kind", "origin_name", "catalog_id"}


class CatalogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    function_spec: FunctionSpec
    keywords: list[str] = Field(default_factory=list)
    provenance: str = ""  # e.g. "derived from workflow X"
    usage_count: int = 0


class UsageRecord(BaseModel):
    """One LLM call's usage + attributed cost, for per-agent run accounting.

    source="tokens": cost derived from token counts × per-model price
    (WCA/FCA/WDA, via LangChain usage_metadata).
    source="sdk": exact provider-computed cost from the Claude Agent SDK
    (FGA's code-generation step). See faasr_agents.pricing.
    """
    agent: str  # "WCA" | "FCA" | "FGA" | "WDA"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    source: Literal["tokens", "sdk"] = "tokens"


class WorkflowEntry(BaseModel):
    """A stored, reusable complete workflow (analogous to CatalogEntry)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_spec: WorkflowSpec
    keywords: list[str] = Field(default_factory=list)
    provenance: str = ""  # e.g. "user request: ..."
    usage_count: int = 0
