from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from faasr_agents import llm, pricing
from faasr_agents.agents import fga
from faasr_agents.models import FunctionSpec


class BackendSelectionTests(unittest.TestCase):
    def test_auto_backend_preserves_existing_provider_behavior(self):
        with (
            patch.object(llm, "selected_fga_backend", "auto"),
            patch.object(llm, "selected_provider", "bedrock"),
        ):
            self.assertEqual(llm.get_fga_backend(), "claude-code")

        with (
            patch.object(llm, "selected_fga_backend", "auto"),
            patch.object(llm, "selected_provider", "openai"),
        ):
            self.assertEqual(llm.get_fga_backend(), "chatopenai")

    def test_explicit_opencode_backend_overrides_provider(self):
        with (
            patch.object(llm, "selected_fga_backend", "opencode"),
            patch.object(llm, "selected_provider", "openai"),
        ):
            self.assertEqual(llm.get_fga_backend(), "opencode")


class OpenCodeConfigurationTests(unittest.TestCase):
    def test_curate_agent_preserves_inline_provider_configuration(self):
        existing = {
            "provider": {
                "ollama": {
                    "options": {"baseURL": "http://127.0.0.1:11434/v1"},
                }
            }
        }
        with patch.dict(
            os.environ,
            {
                "FAASR_OPENCODE_MODEL": "",
                "OPENCODE_CONFIG_CONTENT": json.dumps(existing),
            },
            clear=False,
        ):
            config = json.loads(fga._opencode_config_content())

        self.assertEqual(
            config["provider"]["ollama"]["options"]["baseURL"],
            "http://127.0.0.1:11434/v1",
        )
        agent = config["agent"]["faasr-fga"]
        self.assertEqual(agent["mode"], "primary")
        self.assertEqual(agent["permission"]["edit"], "allow")
        self.assertEqual(agent["permission"]["external_directory"], "deny")

    def test_ollama_model_adds_local_provider_configuration(self):
        with patch.dict(
            os.environ,
            {
                "FAASR_OPENCODE_MODEL": "ollama/curate-ornith:9b",
                "FAASR_OLLAMA_BASE_URL": "http://localhost:11434/v1",
                "OPENCODE_CONFIG_CONTENT": "{}",
            },
            clear=False,
        ):
            config = json.loads(fga._opencode_config_content())

        provider = config["provider"]["ollama"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"]["baseURL"], "http://localhost:11434/v1")
        model = provider["models"]["curate-ornith:9b"]
        self.assertEqual(model["limit"]["context"], 65536)
        self.assertEqual(model["limit"]["output"], 8192)

    def test_subprocess_environment_hides_deployment_secrets(self):
        values = {
            "GH_PAT": "github-secret",
            "S3_AccessKey": "s3-access",
            "S3_SecretKey": "s3-secret",
            "OPENAI_API_KEY": "model-secret",
            "OPENCODE_CONFIG_CONTENT": "{}",
        }
        with patch.dict(os.environ, values, clear=False):
            env = fga._opencode_env()

        self.assertNotIn("GH_PAT", env)
        self.assertNotIn("S3_AccessKey", env)
        self.assertNotIn("S3_SecretKey", env)
        self.assertEqual(env["OPENAI_API_KEY"], "model-secret")
        self.assertIn("faasr-fga", json.loads(env["OPENCODE_CONFIG_CONTENT"])["agent"])

    def test_model_requires_provider_model_format(self):
        with patch.dict(os.environ, {"FAASR_OPENCODE_MODEL": "missing-provider"}):
            with self.assertRaisesRegex(RuntimeError, "provider/model"):
                fga._opencode_model()


class OpenCodeRunnerTests(unittest.TestCase):
    def setUp(self):
        pricing.reset_run_records()

    def tearDown(self):
        pricing.reset_run_records()

    def test_run_uses_selected_model_and_records_reported_usage(self):
        event = {
            "type": "step_finish",
            "part": {
                "tokens": {
                    "input": 100,
                    "output": 20,
                    "cache": {"read": 60, "write": 5},
                },
                "cost": 0.012,
            },
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(event), stderr=""
        )
        spec = FunctionSpec(name="make_plot")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.dict(
                    os.environ,
                    {
                        "FAASR_OPENCODE_MODEL": "ollama/qwen3-coder",
                        "OPENCODE_CONFIG_CONTENT": "{}",
                    },
                    clear=False,
                ),
                patch.object(fga, "_opencode_binary", return_value="/usr/bin/opencode"),
                patch.object(fga.subprocess, "run", return_value=completed) as run,
            ):
                fga._run_opencode_turn(Path(tmpdir), spec, "implement it", 1)

        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/usr/bin/opencode", "run"])
        self.assertEqual(
            command[command.index("--model") + 1],
            "ollama/qwen3-coder",
        )
        self.assertEqual(command[-1], "implement it")

        records = pricing.get_run_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "opencode")
        self.assertEqual(records[0].cache_read_tokens, 60)
        self.assertAlmostEqual(records[0].cost_usd, 0.012)

    def test_timeout_keeps_a_function_that_opencode_already_wrote(self):
        spec = FunctionSpec(name="make_plot")

        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir)
            functions = context_dir / "functions"
            functions.mkdir()

            def timeout_after_write(*args, **kwargs):
                (functions / "make_plot.py").write_text(
                    "def make_plot(folder='workflow_data'):\n    return None\n"
                )
                raise subprocess.TimeoutExpired(cmd=args[0], timeout=600)

            with (
                patch.dict(
                    os.environ,
                    {"FAASR_OPENCODE_MODEL": "ollama/curate-ornith:9b"},
                    clear=False,
                ),
                patch.object(fga, "_opencode_binary", return_value="/usr/bin/opencode"),
                patch.object(fga.subprocess, "run", side_effect=timeout_after_write),
            ):
                fga._run_opencode_turn(context_dir, spec, "implement it", 1)

    def test_timeout_without_a_function_still_fails(self):
        spec = FunctionSpec(name="make_plot")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(fga, "_opencode_binary", return_value="/usr/bin/opencode"),
                patch.object(
                    fga.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired(cmd="opencode", timeout=600),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    fga._run_opencode_turn(Path(tmpdir), spec, "implement it", 1)

    def test_failed_stub_test_is_returned_for_one_repair_attempt(self):
        spec = FunctionSpec(name="make_plot")
        prompts: list[str] = []

        def fake_turn(context_dir, node, prompt, attempt):
            prompts.append(prompt)
            functions = context_dir / "functions"
            functions.mkdir(exist_ok=True)
            (functions / f"{node.name}.py").write_text(
                "def make_plot(folder='workflow_data'):\n    return None\n"
            )
            (functions / f"{node.name}.deps.txt").write_text("")

        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir)
            with (
                patch.dict(os.environ, {"FAASR_OPENCODE_MODEL": ""}, clear=False),
                patch.object(fga, "_run_opencode_turn", side_effect=fake_turn) as run,
                patch.object(
                    fga,
                    "_run_stub_test",
                    side_effect=[(False, "AssertionError: wrong output"), (True, "")],
                ),
            ):
                result = fga._implement_all_opencode([spec], context_dir)

        self.assertEqual(run.call_count, 2)
        self.assertIn("AssertionError: wrong output", prompts[1])
        self.assertIn("def make_plot", result[0].code or "")


if __name__ == "__main__":
    unittest.main()
