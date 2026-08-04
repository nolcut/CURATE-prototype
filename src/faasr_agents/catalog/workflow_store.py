from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from faasr_agents.models import WorkflowEntry, CATALOG_EXCLUDED_SPEC_FIELDS

_DEFAULT_DATA_DIR = Path(__file__).parent / "workflows"


class WorkflowRegistry:
    """Stores complete, reusable workflows — sibling of CatalogStore (functions)."""

    def __init__(self, data_dir: Path = _DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, WorkflowEntry] = {}
        self._load_all()

    # ── persistence ──────────────────────────────────────────────────────────

    def _entry_path(self, entry_id: str) -> Path:
        return self.data_dir / f"{entry_id}.json"

    def _load_all(self) -> None:
        for p in self.data_dir.glob("*.json"):
            try:
                entry = WorkflowEntry.model_validate(json.loads(p.read_text()))
                self._entries[entry.id] = entry
            except Exception:
                pass

    def _save(self, entry: WorkflowEntry) -> None:
        self._entry_path(entry.id).write_text(
            entry.model_dump_json(
                indent=2,
                exclude={"workflow_spec": {"nodes": {"__all__": CATALOG_EXCLUDED_SPEC_FIELDS}}},
            )
        )

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, entry: WorkflowEntry) -> WorkflowEntry:
        self._entries[entry.id] = entry
        self._save(entry)
        return entry

    def get(self, entry_id: str) -> Optional[WorkflowEntry]:
        return self._entries.get(entry_id)

    def get_by_name(self, name: str) -> Optional[WorkflowEntry]:
        for entry in self._entries.values():
            if entry.workflow_spec.name == name:
                return entry
        return None

    def increment_usage(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.usage_count += 1
            self._save(entry)

    def list_all(self) -> list[WorkflowEntry]:
        return list(self._entries.values())
