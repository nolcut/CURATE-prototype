from __future__ import annotations
import logging
from typing import Optional

from FaaSr_py.helpers.graph_functions import validate_json, check_dag


def validate_faasr_json(workflow_dict: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a FaaSr workflow dict against the FaaSr schema and check for DAG issues.
    Uses FaaSr_py's validate_json (schema check) and check_dag (cycle/reachability check).
    Returns (is_valid, error_message_or_None).
    """
    faasr_logger = logging.getLogger("FaaSr_py.helpers.graph_functions")
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(self.format(record))

    handler = _Capture()
    faasr_logger.addHandler(handler)
    try:
        try:
            validate_json(workflow_dict)
            check_dag(workflow_dict)
        except SystemExit as e:
            return False, "\n".join(captured) or f"Validation failed (exit {e.code})"
        return True, None
    finally:
        faasr_logger.removeHandler(handler)
