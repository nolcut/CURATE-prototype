"""Workflow output artifact helpers shared by the Gate-5 REPL and the export bundle.

Artifacts are the workflow's S3 outputs: every `output<N>` argument in the
emitted ActionList, keyed under `<folder>/<filename>`. Ranked outputs carry a
`{rank}` placeholder that is resolved against the real objects in S3.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def init_s3(workflow_json: dict):
    """Build a FaaSrS3Client from env creds + the workflow JSON's DataStores.

    Returns None (after printing the reason) if S3 is unavailable — callers
    treat downloads as disabled.
    """
    try:
        from faasr_agents.faasr.runtime.s3_client import FaaSrS3Client
        return FaaSrS3Client(
            workflow_data=workflow_json,
            access_key=os.getenv("S3_AccessKey", ""),
            secret_key=os.getenv("S3_SecretKey", ""),
        )
    except Exception as e:
        print(f"  (S3 not available — download/open disabled: {e})")
        return None


def build_artifact_list(workflow_json: dict) -> list[dict]:
    artifacts = []
    for action_name, action_def in workflow_json.get("ActionList", {}).items():
        args = action_def.get("Arguments", {})
        folder = args.get("folder", "")
        for k in sorted(args):
            if k.startswith("output") and k[6:].isdigit():
                v = args[k]
                ext = Path(v).suffix.lstrip(".").lower()
                artifacts.append({
                    "index": len(artifacts),
                    "action": action_name,
                    "filename": v,
                    "s3_key": f"{folder}/{v}" if folder else v,
                    "format": ext or "?",
                })
    return artifacts


def expand_ranked_artifacts(artifacts: list[dict], s3) -> list[dict]:
    """Resolve `{rank}` placeholders into concrete, downloadable artifacts.

    A ranked artifact's filename carries a `{rank}` placeholder (e.g.
    `shuffle_shard_{rank}.json`). The number of shards isn't statically the
    producing node's own rank (an unranked producer may shard across a ranked
    successor), so we enumerate ground truth: list the matching objects in S3
    and emit one concrete entry per real file. Non-ranked artifacts pass
    through. If S3 is unavailable or nothing matches, the template entry is
    kept so the listing isn't silently empty.
    """
    expanded: list[dict] = []
    for a in artifacts:
        key = a.get("s3_key", "")
        if "{rank}" not in key:
            expanded.append(a)
            continue

        found: list[str] = []
        if s3 is not None:
            prefix = key.split("{rank}", 1)[0]
            pattern = re.compile(
                "^" + re.escape(key).replace(re.escape("{rank}"), r"(\d+)") + "$"
            )
            try:
                matches = [(m, int(m.group(1))) for k in s3.list_objects(prefix)
                           if (m := pattern.match(k))]
                found = [m.string for m, _ in sorted(matches, key=lambda t: t[1])]
            except Exception:
                found = []

        if not found:
            expanded.append(a)  # leave template; download will surface a clear error
            continue

        for k in found:
            expanded.append({**a, "s3_key": k, "filename": k.rsplit("/", 1)[-1]})

    for i, a in enumerate(expanded):
        a["index"] = i
    return expanded
