"""Per-run evaluation bundle, written by the Gate-5 REPL `export <dir>` command.

Writes a self-contained, navigable artifact bundle so the SeBS-Flow experiments
can be evaluated from disk: the verbatim prompt, the workflow spec +
per-function code, token costs (CSV), per-gate revision timeline, FCA
sourcing/adaptation decisions, and the full run log.

Events and console output are accumulated in module globals (mirroring
pricing._RUN_RECORDS) because Gate-1/Gate-2 revision rounds happen inside
interrupt()-replay loops that never reach graph state — the CLI records them
as they occur, unconditionally, so an export requested at Gate 5 has the
whole run.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_EVENTS: list[dict] = []

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def reset_events() -> None:
    _EVENTS.clear()


def record_event(kind: str, **fields) -> None:
    """Append one timestamped event (gate shown/decided, fca decisions, ...)."""
    _EVENTS.append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        **fields,
    })


def get_events() -> list[dict]:
    return list(_EVENTS)


_CONSOLE = io.StringIO()


class _Tee(io.TextIOBase):
    """Duplicate a stream into the in-memory console buffer (ANSI stripped)."""

    def __init__(self, stream, buffer):
        self._stream = stream
        self._buffer = buffer

    def write(self, s: str) -> int:
        n = self._stream.write(s)
        self._buffer.write(_ANSI_RE.sub("", s))
        return n

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def install_console_capture() -> None:
    """Mirror stdout+stderr into the in-memory buffer for the rest of the run."""
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, _CONSOLE)
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr, _CONSOLE)


def get_console_text() -> str:
    return _CONSOLE.getvalue()


def make_run_dir(export_dir: str | os.PathLike) -> Path:
    """Create <export_dir>/run_<timestamp>/ (and export_dir itself if missing)."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(export_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "").strip("_") or "workflow"


def _gate_rows(events: list[dict]) -> list[dict]:
    """Flatten gate events into timeline rows for gates.csv."""
    rows = []
    for i, ev in enumerate(events, 1):
        if ev["kind"] not in ("gate_shown", "gate_decision"):
            continue
        rows.append({
            "seq": i,
            "at": ev["at"],
            "gate": ev.get("gate", ""),
            "event": "shown" if ev["kind"] == "gate_shown" else ev.get("action", ""),
            "directive": (ev.get("directive") or "").replace("\n", " "),
        })
    return rows


def _revisions_by_gate(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in events:
        if ev["kind"] == "gate_decision" and ev.get("action") == "revise":
            key = f"gate{ev.get('gate', '?')}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_run(
    run_dir: Path,
    final_state: dict,
    thread_id: str,
    records: list,
    events: list[dict],
    meta: dict,
) -> Path:
    """Write the full evaluation bundle into run_dir. Returns the bundle path.

    Never raises for partial runs: sections whose data is absent (no spec, no
    deploy) are skipped or stubbed so a failed run still exports cleanly.
    """
    from faasr_agents import pricing

    run_dir = Path(run_dir)
    spec = final_state.get("workflow_spec")
    deploy_result = final_state.get("deploy_result") or {}
    user_request = (final_state.get("user_request") or "").strip()
    workflow_name = getattr(spec, "name", None) or "workflow"
    now = datetime.now().isoformat(timespec="seconds")

    for sub in ("workflow", "workflow/functions", "costs", "revisions", "adaptation", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── prompt.txt ────────────────────────────────────────────────────────────
    if user_request:
        (run_dir / "prompt.txt").write_text(user_request + "\n")
    else:
        (run_dir / "prompt.txt").write_text(
            f"(no prompt — reused stored workflow '{workflow_name}', deploy-only run)\n"
        )

    # ── workflow/ ─────────────────────────────────────────────────────────────
    sourcing_rows: list[dict] = []
    workflow_json: dict | None = None
    if spec is not None:
        (run_dir / "workflow" / "spec.json").write_text(
            json.dumps(spec.model_dump(), indent=2, default=str) + "\n"
        )
        try:
            from faasr_agents.faasr.emit import emit_faasr_json
            workflow_json = emit_faasr_json(spec)
            (run_dir / "workflow" / "workflow.json").write_text(
                json.dumps(workflow_json, indent=2, default=str) + "\n"
            )
        except Exception as e:
            (run_dir / "workflow" / "workflow.json").write_text(
                json.dumps({"error": f"could not emit FaaSr JSON: {e}"}, indent=2) + "\n"
            )
        for node in spec.nodes:
            if node.code:
                (run_dir / "workflow" / "functions" / f"{node.name}.py").write_text(node.code)
            sourcing_rows.append({
                "name": node.name,
                "source": node.source,
                "origin_kind": node.origin_kind or "",
                "origin_name": node.origin_name or "",
                "catalog_id": node.catalog_id or "",
                "implemented": bool(node.code),
            })

    # ── costs/ ────────────────────────────────────────────────────────────────
    record_dicts = [r.model_dump() for r in records]
    _write_csv(
        run_dir / "costs" / "records.csv", record_dicts,
        ["agent", "model", "input_tokens", "output_tokens",
         "cache_read_tokens", "cache_write_tokens", "cost_usd", "source"],
    )
    by_agent, total = pricing.summarize_run(records) if records else ({}, {})
    summary_rows = [
        {"agent": name, **stats} for name, stats in sorted(by_agent.items())
    ]
    if total:
        summary_rows.append({"agent": "TOTAL", **total})
    _write_csv(
        run_dir / "costs" / "summary.csv", summary_rows,
        ["agent", "calls", "input_tokens", "output_tokens",
         "cache_read_tokens", "cache_write_tokens", "cost_usd"],
    )

    # ── revisions/ ────────────────────────────────────────────────────────────
    gate_rows = _gate_rows(events)
    _write_csv(
        run_dir / "revisions" / "gates.csv", gate_rows,
        ["seq", "at", "gate", "event", "directive"],
    )
    directives = final_state.get("revision_directives") or []
    (run_dir / "revisions" / "directives.txt").write_text(
        "\n".join(directives) + ("\n" if directives else "")
    )

    # ── adaptation/ ───────────────────────────────────────────────────────────
    fca_rounds = [
        {"at": ev["at"], "summary": ev.get("summary")}
        for ev in events if ev["kind"] == "fca_decisions"
    ]
    (run_dir / "adaptation" / "fca_decisions.json").write_text(
        json.dumps(fca_rounds, indent=2, default=str) + "\n"
    )
    _write_csv(
        run_dir / "adaptation" / "sourcing.csv", sourcing_rows,
        ["name", "source", "origin_kind", "origin_name", "catalog_id", "implemented"],
    )

    # ── logs/ ─────────────────────────────────────────────────────────────────
    logs = deploy_result.get("logs", "")
    (run_dir / "logs" / "execution.txt").write_text(
        logs if logs and logs != "(skipped)" else "(deploy/execution did not run)\n"
    )
    console = get_console_text()
    (run_dir / "logs" / "console.txt").write_text(
        "(console output up to the moment of export)\n\n" + console
        if console else "(no console output captured)\n"
    )
    with open(run_dir / "logs" / "events.jsonl", "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")

    # ── artifacts/ ────────────────────────────────────────────────────────────
    artifact_manifest = _download_artifacts(run_dir, workflow_json, deploy_result)

    # ── summary.json ──────────────────────────────────────────────────────────
    revisions_by_gate = _revisions_by_gate(events)
    summary = {
        "workflow": workflow_name,
        "thread_id": thread_id,
        "exported_at": now,
        **meta,
        "success": deploy_result.get("success"),
        "failure_class": deploy_result.get("failure_class"),
        "iteration": final_state.get("iteration", 0),
        "revisions_by_gate": revisions_by_gate,
        "revisions_total": sum(revisions_by_gate.values()),
        "nodes": sourcing_rows,
        "artifacts_downloaded": sum(1 for a in artifact_manifest if a.get("downloaded")),
        "artifacts_total": len(artifact_manifest),
        "cost_total": total,
        "cost_by_agent": by_agent,
        # Manual verdict — fill in after evaluating behavior correctness:
        # "GT" (ground-truth match), "HV" (human verified), or "Fail".
        "behavior_verdict": None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    # ── evaluation.md ─────────────────────────────────────────────────────────
    (run_dir / "evaluation.md").write_text(_evaluation_md(
        workflow_name, user_request, spec, deploy_result,
        by_agent, total, revisions_by_gate, final_state.get("iteration", 0), now,
    ))

    # ── README.md ─────────────────────────────────────────────────────────────
    (run_dir / "README.md").write_text(_readme_md(workflow_name, now))

    # Rename run_<ts> → <workflow> (then <workflow>-1, -2, … when re-exporting
    # to the same folder) now that the name is known. Guarded on the run_
    # prefix so re-exporting an already-renamed dir doesn't stack names.
    if run_dir.name.startswith("run_"):
        base = _safe_name(workflow_name)
        target = run_dir.parent / base
        n = 0
        while target.exists():
            n += 1
            target = run_dir.parent / f"{base}-{n}"
        try:
            run_dir = run_dir.rename(target)
        except OSError:
            pass
    return run_dir


def _download_artifacts(run_dir: Path, workflow_json: dict | None,
                        deploy_result: dict) -> list[dict]:
    """Download the workflow's S3 output artifacts into artifacts/.

    Writes artifacts/manifest.json describing every expected output (action,
    filename, s3_key, downloaded, error) so the bundle explains itself even
    when S3 is unreachable. Returns the manifest rows; never raises. Skipped
    entirely (no artifacts/ dir) when the workflow never ran.
    """
    logs = deploy_result.get("logs", "")
    if workflow_json is None or not logs or logs == "(skipped)":
        return []

    manifest: list[dict] = []
    try:
        from faasr_agents.faasr.artifacts import (
            build_artifact_list, expand_ranked_artifacts, init_s3,
        )

        out_dir = run_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        s3 = init_s3(workflow_json)
        artifacts = expand_ranked_artifacts(build_artifact_list(workflow_json), s3)

        for a in artifacts:
            row = {
                "action": a.get("action", ""),
                "filename": a.get("filename", ""),
                "s3_key": a.get("s3_key", ""),
                "downloaded": False,
                "error": None,
            }
            if s3 is None:
                row["error"] = "S3 client unavailable (missing credentials or DataStores config)"
            else:
                try:
                    s3.download_object(a["s3_key"], str(out_dir / a["filename"]))
                    row["downloaded"] = True
                except Exception as e:
                    row["error"] = str(e)
            manifest.append(row)

        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
    except Exception as e:
        print(f"  (artifact download failed: {e})")
    return manifest


def _evaluation_md(workflow_name, user_request, spec, deploy_result,
                   by_agent, total, revisions_by_gate, iteration, now) -> str:
    lines = [
        f"# Evaluation — {workflow_name}",
        "",
        f"Exported: {now}",
        "",
        "## Prompt",
        "",
        "```",
        user_request or "(reused stored workflow — no prompt)",
        "```",
        "",
        "## Qualitative",
        "",
        "### Workflow structure (precedence / skeleton)",
        "",
    ]
    if spec is not None:
        lines.append("Nodes:")
        for n in spec.nodes:
            rank = f"  (rank {n.rank})" if getattr(n, "rank", 1) > 1 else ""
            lines.append(f"- `{n.name}`{rank} — {n.description or '(no description)'}")
        lines += ["", "Edges:"]
        for e in spec.edges:
            lines.append(f"- `{e.from_node}` → `{e.to_node}`")
        lines += [
            "",
            "- [ ] Structure is correct",
            "",
            "### Node sourcing",
            "",
            "| node | source | origin | catalog_id |",
            "|---|---|---|---|",
        ]
        for n in spec.nodes:
            origin = f"{n.origin_kind}:{n.origin_name}" if n.origin_kind else ""
            lines.append(f"| `{n.name}` | {n.source} | {origin} | {n.catalog_id or ''} |")
        lines += [
            "",
            "- [ ] Nodes are sourced from the appropriate locations (new / cache / catalog)",
            "- [ ] Correct nodes were selected for adaptation",
        ]
    else:
        lines.append("(no workflow spec — run ended before composition)")
    lines += [
        "",
        "## Quantitative",
        "",
        "### Token cost",
        "",
        "| agent | calls | in tok | out tok | cost (USD) |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(by_agent):
        a = by_agent[name]
        lines.append(
            f"| {name} | {a['calls']} | {a['input_tokens']:,} "
            f"| {a['output_tokens']:,} | ${a['cost_usd']:.4f} |"
        )
    if total:
        lines.append(
            f"| **TOTAL** | {total['calls']} | {total['input_tokens']:,} "
            f"| {total['output_tokens']:,} | **${total['cost_usd']:.4f}** |"
        )
    lines += [
        "",
        "### Revisions",
        "",
        f"- Per gate: {json.dumps(revisions_by_gate) if revisions_by_gate else 'none'}",
        f"- Deploy/output revision cycles (`iteration`): {iteration}",
        "",
        "### Deploy outcome",
        "",
        f"- success: {deploy_result.get('success')}",
        f"- failure_class: {deploy_result.get('failure_class')}",
        "",
        "## Behavior correctness (manual)",
        "",
        "Compare outputs against the reference (verbatim for SeBS-Flow MapReduce;",
        "within tolerance for ML examples; by hand otherwise), then record:",
        "",
        "- Verdict: `[ ]` GT (ground-truth)  `[ ]` HV (human verified)  `[ ]` Fail",
        "- Method:",
        "- Notes:",
        "",
        "Also set `behavior_verdict` in `summary.json` (\"GT\" | \"HV\" | \"Fail\").",
        "",
        "_Tip: run with `--debug` to also keep per-function FGA agent traces_",
        "_(path printed during the run; not copied into this bundle)._",
        "",
    ]
    return "\n".join(lines)


def _readme_md(workflow_name, now) -> str:
    return f"""# Run bundle — {workflow_name}

Exported {now} by `faasr-agents --export`.

| Path | Contents |
|---|---|
| `prompt.txt` | Verbatim user request |
| `summary.json` | Machine-readable rollup (cost, revisions, sourcing, outcome; fill in `behavior_verdict`) |
| `evaluation.md` | Answer sheet for the evaluation questions (structure, sourcing, cost, verdict) |
| `workflow/spec.json` | Full workflow spec incl. per-node source/origin provenance |
| `workflow/workflow.json` | Emitted FaaSr payload |
| `workflow/functions/` | Generated per-function Python code |
| `artifacts/` | Workflow output files downloaded from S3 (+ `manifest.json` with per-file status) |
| `costs/records.csv` | Every LLM call: agent, model, tokens, cost |
| `costs/summary.csv` | Per-agent rollup + total |
| `revisions/gates.csv` | Gate timeline: every gate shown and every decision, with directives |
| `revisions/directives.txt` | All explicit user change requests |
| `adaptation/fca_decisions.json` | FCA sourcing decisions per resolve round |
| `adaptation/sourcing.csv` | Final per-node sourcing (new / cache / catalog / adapt) |
| `logs/console.txt` | Console output of the run (up to the moment of export) |
| `logs/execution.txt` | Deployed-run execution logs (per-function blocks) |
| `logs/events.jsonl` | Structured event log |
"""
