from __future__ import annotations

import unittest

from faasr_agents.faasr.emit import emit_faasr_json
from faasr_agents.faasr.validate import validate_faasr_json
from faasr_agents.models import FunctionSpec, IOSpec, WorkflowSpec


class FaaSrValidationTests(unittest.TestCase):
    def test_single_terminal_action_is_a_valid_dag(self):
        spec = WorkflowSpec(
            name="local_function_test",
            entry="calculate_number_summary",
            nodes=[
                FunctionSpec(
                    name="calculate_number_summary",
                    inputs=[IOSpec(name="run_params.json", type="json")],
                    outputs=[IOSpec(name="number_summary.json", type="json")],
                )
            ],
        )

        workflow = emit_faasr_json(spec)

        self.assertEqual(workflow["FunctionInvoke"], "calculate-number-summary")
        self.assertEqual(
            workflow["ActionList"]["calculate-number-summary"]["InvokeNext"], []
        )
        self.assertEqual(validate_faasr_json(workflow), (True, None))

    def test_single_action_still_requires_a_valid_entry(self):
        workflow = emit_faasr_json(
            WorkflowSpec(
                name="local_function_test",
                entry="calculate_number_summary",
                nodes=[FunctionSpec(name="calculate_number_summary")],
            )
        )
        workflow["FunctionInvoke"] = "missing-action"

        valid, error = validate_faasr_json(workflow)

        self.assertFalse(valid)
        self.assertIn("FunctionInvoke", error or "")


if __name__ == "__main__":
    unittest.main()
