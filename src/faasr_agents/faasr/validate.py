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
            actions = workflow_dict.get("ActionList") or {}
            entry = workflow_dict.get("FunctionInvoke")
            only_action = next(iter(actions.values()), {}) if len(actions) == 1 else {}
            invoke_next = only_action.get("InvokeNext", [])

            # FaaSr_py's graph builder records nodes only while traversing edges,
            # so check_dag raises KeyError for the valid one-node/no-edge case.
            if len(actions) == 1 and not invoke_next:
                if entry not in actions:
                    return False, "FunctionInvoke does not refer to a valid function"
            else:
                check_dag(workflow_dict)
        except SystemExit as e:
            return False, "\n".join(captured) or f"Validation failed (exit {e.code})"
        except Exception as e:
            detail = "\n".join(captured)
            return False, detail or f"Validation failed ({type(e).__name__}: {e})"
        return True, None
    finally:
        faasr_logger.removeHandler(handler)
