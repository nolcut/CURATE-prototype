from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi
from faasr_agents.models import CatalogEntry, CATALOG_EXCLUDED_SPEC_FIELDS

_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


class CatalogStore:
    def __init__(self, data_dir: Path = _DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CatalogEntry] = {}
        self._load_all()

    # ── persistence ──────────────────────────────────────────────────────────

    def _entry_path(self, entry_id: str) -> Path:
        return self.data_dir / f"{entry_id}.json"

    def _load_all(self) -> None:
        for p in self.data_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                # Recover code from companion .py if JSON has none (migration path)
                if not data.get("function_spec", {}).get("code"):
                    py_file = p.with_suffix(".py")
                    if py_file.exists():
                        raw = py_file.read_text()
                        lines = raw.splitlines()
                        start = next((i for i, l in enumerate(lines) if not l.startswith("#")), 0)
                        data["function_spec"]["code"] = "\n".join(lines[start:]).strip()
                entry = CatalogEntry.model_validate(data)
                self._entries[entry.id] = entry
            except Exception:
                pass

    def _save(self, entry: CatalogEntry) -> None:
        self._entry_path(entry.id).write_text(
            entry.model_dump_json(
                indent=2,
                exclude={"function_spec": CATALOG_EXCLUDED_SPEC_FIELDS},
            )
        )

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, entry: CatalogEntry) -> CatalogEntry:
        self._entries[entry.id] = entry
        self._save(entry)
        return entry

    def get(self, entry_id: str) -> Optional[CatalogEntry]:
        return self._entries.get(entry_id)

    def increment_usage(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.usage_count += 1
            self._save(entry)

    def search(self, query: str, k: int = 5) -> list[CatalogEntry]:
        if not self._entries:
            return []

        entries = list(self._entries.values())
        corpus = []
        for e in entries:
            spec = e.function_spec
            tokens = (
                spec.name.split("_")
                + spec.description.lower().split()
                + e.keywords
                + [inp.name for inp in spec.inputs]
                + [out.name for out in spec.outputs]
            )
            corpus.append(tokens)

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, entries), key=lambda x: x[0], reverse=True)
        return [entry for score, entry in ranked[:k] if score > 0]

    def list_all(self) -> list[CatalogEntry]:
        return list(self._entries.values())
