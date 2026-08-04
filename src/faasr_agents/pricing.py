"""Per-model price table and cost computation for agent LLM usage.

Two cost sources feed the run accounting:

* LangChain `llm.invoke(...)` responses expose token counts via
  `response.usage_metadata` (populated by both ChatAnthropic and
  ChatBedrockConverse) but no dollar amount — we derive cost as tokens × the
  per-model rate below. This is an estimate: it is exact only insofar as the
  rate table matches the provider actually billed (Anthropic vs. Bedrock are
  priced the same per token for these models) and prices haven't drifted.
* The Claude Agent SDK (`agents/fga.py`) returns an exact provider-computed
  `total_cost_usd` — that path uses `record_sdk_usage` and does not touch this
  table.

Normalization invariant: UsageRecord.input_tokens is ALWAYS the cache-inclusive
total input (uncached + cache reads + cache writes); the cache_read_tokens /
cache_write_tokens columns are the breakdown. Provider integrations differ
(langchain_anthropic reports the inclusive total, langchain_aws and the Agent
SDK report the uncached count only), so record_usage/record_sdk_usage normalize
at record time.

Prices are USD per 1M tokens. Update as list prices change. Sources:
Anthropic model pricing (Opus 4.8 $5/$25, Sonnet 4.5/4.6 $3/$15,
Haiku 4.5 $1/$5, legacy Claude 3.5 Haiku $0.80/$4). Prompt-cache multipliers
follow Anthropic's published economics: cache reads ~0.1× input, cache writes
1.25× input (5-min TTL) / 2× input (1-hour TTL). We assume the 5-min write
rate since that is the default TTL used by the agents.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

_PER_MTOK = 1_000_000.0

# (input_per_mtok, output_per_mtok), keyed by a coarse model tier. The tier is
# resolved by substring so one entry covers both the Anthropic ID and the
# Bedrock cross-region profile (e.g. "claude-opus-4-8" and
# "us.anthropic.claude-opus-4-8-...").
_TIERS: dict[str, tuple[float, float]] = {
    "opus":         (5.0, 25.0),
    "sonnet":       (3.0, 15.0),
    "haiku-4-5":    (1.0, 5.0),
    "3-5-haiku":    (0.80, 4.0),   # legacy Claude 3.5 Haiku (Bedrock fast tier)
    "3-haiku":      (0.80, 4.0),
}

_CACHE_READ_MULT = 0.10
_CACHE_WRITE_MULT = 1.25


# Process-scoped accumulator of every UsageRecord produced this run. A module
# global (not AgentState) is deliberate: LangGraph re-runs a node from the top on
# each interrupt() resume, so Gate-1/Gate-2 revision rounds fire real API calls
# whose cost the paused execution never returns via state. Recording at the
# moment of each invoke captures ALL calls — revisions included — exactly. The
# CLI runs one workflow per process; it calls reset_run_records() at run start
# and get_run_records() at the end. A long-lived/multi-run host must reset per
# run (and would cross-contaminate under concurrent runs — not a concern here).
_RUN_RECORDS: list = []


def reset_run_records() -> None:
    """Clear the run accumulator. Call once at the start of each run."""
    _RUN_RECORDS.clear()


def get_run_records() -> list:
    """Return all UsageRecords accumulated since the last reset."""
    return list(_RUN_RECORDS)


def summarize_run(records: list) -> tuple[dict, dict]:
    """Roll UsageRecords up into (by_agent, total).

    by_agent: {agent: {calls, input_tokens, output_tokens,
                       cache_read_tokens, cache_write_tokens, cost_usd}}
    total:    same keys, across all records
    """
    by_agent: dict[str, dict] = {}
    for r in records:
        a = by_agent.setdefault(
            r.agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0}
        )
        a["calls"] += 1
        a["input_tokens"] += r.input_tokens
        a["output_tokens"] += r.output_tokens
        a["cache_read_tokens"] += r.cache_read_tokens
        a["cache_write_tokens"] += r.cache_write_tokens
        a["cost_usd"] += r.cost_usd
    total = {
        "calls": len(records),
        "input_tokens": sum(r.input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "cache_read_tokens": sum(r.cache_read_tokens for r in records),
        "cache_write_tokens": sum(r.cache_write_tokens for r in records),
        "cost_usd": sum(r.cost_usd for r in records),
    }
    return by_agent, total


def _tier_for(model: str) -> tuple[float, float] | None:
    m = (model or "").lower()
    # Order matters: match the most specific haiku variants before the generic.
    if "opus" in m:
        return _TIERS["opus"]
    if "sonnet" in m:
        return _TIERS["sonnet"]
    if "3-5-haiku" in m or "3.5-haiku" in m:
        return _TIERS["3-5-haiku"]
    if "3-haiku" in m:
        return _TIERS["3-haiku"]
    if "haiku" in m:
        return _TIERS["haiku-4-5"]
    return None


def cost_from_usage(model: str, usage_metadata: dict | None) -> float:
    """Compute USD cost from a LangChain `usage_metadata` dict.

    `usage_metadata` follows LangChain's normalized shape: `input_tokens` is the
    TOTAL input (cache reads/creation included), broken down in
    `input_token_details`. We back out the uncached portion so cache tokens are
    billed at their own (cheaper) rate rather than full input rate.

    Returns 0.0 (and logs) for an unknown model — callers still record the token
    counts, so the miss is visible rather than silently mispriced.
    """
    if not usage_metadata:
        return 0.0
    rates = _tier_for(model)
    if rates is None:
        log.warning("No price entry for model %r; usage recorded at $0.00", model)
        return 0.0
    in_rate, out_rate = rates

    input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
    output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
    details = usage_metadata.get("input_token_details") or {}
    cache_read = int(details.get("cache_read", 0) or 0)
    cache_write = int(details.get("cache_creation", 0) or 0)

    uncached_input = max(0, input_tokens - cache_read - cache_write)

    cost = (
        uncached_input * in_rate
        + cache_read * in_rate * _CACHE_READ_MULT
        + cache_write * in_rate * _CACHE_WRITE_MULT
        + output_tokens * out_rate
    ) / _PER_MTOK
    return cost


def record_usage(response, agent: str):
    """Build a UsageRecord (source='tokens') from a LangChain response.

    Reads token counts from `response.usage_metadata` and resolves the model
    from `response.response_metadata` (ChatAnthropic sets `model_name`,
    ChatBedrockConverse sets `model_id`).
    """
    from faasr_agents.models import UsageRecord

    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", None) or {}
    model = meta.get("model_name") or meta.get("model_id") or meta.get("model") or ""

    details = usage.get("input_token_details") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cache_read = int(details.get("cache_read", 0) or 0)
    cache_write = int(details.get("cache_creation", 0) or 0)
    # Normalize input_tokens to the cache-INCLUSIVE total. langchain_anthropic
    # already folds cache tokens into input_tokens; langchain_aws (Bedrock
    # Converse — recognizable by the AWS response envelope) reports only the
    # uncached count, so add the cache tokens back for a consistent meaning.
    if "ResponseMetadata" in meta:
        input_tokens += cache_read + cache_write
        usage = {**usage, "input_tokens": input_tokens}

    rec = UsageRecord(
        agent=agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=cost_from_usage(model, usage),
        source="tokens",
    )
    _RUN_RECORDS.append(rec)
    return rec


def record_sdk_usage(result, agent: str, model: str = ""):
    """Build a UsageRecord (source='sdk') from a Claude Agent SDK ResultMessage.

    `total_cost_usd` is exact (provider-computed); token counts come from the
    SDK `usage` dict when present (shape varies, so we read defensively).
    """
    from faasr_agents.models import UsageRecord

    usage = getattr(result, "usage", None) or {}
    if not isinstance(usage, dict):
        usage = {}

    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    rec = UsageRecord(
        agent=agent,
        model=model,
        # SDK usage reports the uncached input only; normalize to the
        # cache-inclusive total (the cache columns keep the breakdown).
        input_tokens=int(usage.get("input_tokens", 0) or 0) + cache_read + cache_write,
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cost_usd=float(getattr(result, "total_cost_usd", 0.0) or 0.0),
        source="sdk",
    )
    _RUN_RECORDS.append(rec)
    return rec
