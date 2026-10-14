"""Validate portable presets and override precedence without loading a model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dunlin.config import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_CONTEXT_LENGTHS,
    describe_config,
    resolve_config,
    resolve_output_root,
    resolve_revisions,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_defaults_preset_and_cli_precedence(self) -> None:
        defaults = resolve_config(None, {})
        self.assertEqual(defaults.context_lengths, DEFAULT_CONTEXT_LENGTHS)
        self.assertEqual(defaults.batch_sizes, DEFAULT_BATCH_SIZES)
        self.assertEqual(defaults.warmup_runs_per_phase, 5)
        self.assertEqual(defaults.measured_runs_per_phase, 30)
        dry_run = describe_config(defaults, "batch_context_matrix", "cpu", "torch.float32")
        matrix_points = dry_run["matrix_points"]
        assert isinstance(matrix_points, list)
        self.assertEqual(len(matrix_points), 20)

        preset = ROOT / "configs" / "smoke_forward_phases.toml"
        settings = resolve_config(preset, {"device": "mps", "measured_runs_per_phase": 3})
        self.assertEqual(settings.context_lengths, (32,))
        self.assertEqual(settings.batch_sizes, (1,))
        self.assertEqual(settings.device, "mps")
        self.assertEqual(settings.dtype, "float32")
        self.assertEqual(settings.warmup_runs_per_phase, 1)
        self.assertEqual(settings.measured_runs_per_phase, 3)
        self.assertEqual(settings.model_revision, "7ae557604adf67be50417f59c2c2f167def9a775")

    def test_rejects_unknown_keys_and_invalid_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.toml"
            path.write_text("schema_version = 1\nunknown = 5\n")
            with self.assertRaisesRegex(ValueError, "Unknown preset keys"):
                resolve_config(path, {})
            path.write_text('schema_version = 1\nmodel_id = "Example/Another-Qwen2"\n')
            self.assertEqual(resolve_config(path, {}).model_id, "Example/Another-Qwen2")
        for overrides, message in (
            ({"model_id": "  "}, "model_id must be a nonempty string"),
            ({"model_id": 5}, "model_id must be a nonempty string"),
            ({"context_lengths": [64, 32]}, "strictly increasing"),
            ({"batch_sizes": [1, 1]}, "strictly increasing"),
            ({"batch_sizes": [0]}, "positive integers"),
            ({"warmup_runs_per_phase": -1}, "warmup_runs_per_phase"),
            ({"measured_runs_per_phase": 0}, "measured_runs_per_phase"),
            ({"phases": ["cached_decode", "prefill"]}, "phases"),
            ({"decode_tokens_per_request": 32}, "decode_tokens_per_request"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                resolve_config(None, overrides)

    def test_forward_runner_rejects_an_incompatible_model_type(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "experiments"))
        from batched_forward_phases import run_experiment

        settings = resolve_config(
            None, {"model_id": "Example/Other-Architecture"},
            default_context_lengths=(32,), default_batch_sizes=(1,),
        )
        with (
            patch("batched_forward_phases.SourceCheckout"),
            patch("batched_forward_phases.select_device", return_value=("cpu", None)),
            patch(
                "forward_workload.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(model_type="llama"),
            ),
            patch("forward_workload.Qwen2ForCausalLM.from_pretrained") as load_model,
            self.assertRaisesRegex(ValueError, "requires a Qwen2 causal LM"),
        ):
            run_experiment(settings, ROOT / "runs" / "unused", {})
        load_model.assert_not_called()

    def test_shared_workload_metadata_records_identity(self) -> None:
        import sys
        import torch
        from transformers import BatchEncoding, PreTrainedTokenizerBase, Qwen2ForCausalLM

        sys.path.insert(0, str(ROOT / "experiments"))
        from forward_workload import workload_metadata

        settings = resolve_config(
            None, {"model_revision": "a" * 40, "tokenizer_revision": "b" * 40},
            default_context_lengths=(500,), default_batch_sizes=(8,),
        )
        model = cast(Qwen2ForCausalLM, SimpleNamespace(
            config=SimpleNamespace(
                _commit_hash="a" * 40, _attn_implementation="sdpa",
                num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=16,
            ),
        ))
        tokenizer = cast(PreTrainedTokenizerBase, SimpleNamespace(
            init_kwargs={"_commit_hash": "b" * 40},
        ))
        inputs = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
        record = workload_metadata(settings, model, tokenizer, "formatted", inputs, "cuda:0", torch.float16)

        self.assertEqual(record["model_revision"], "a" * 40)
        self.assertEqual(record["tokenizer_revision"], "b" * 40)
        self.assertEqual(record["batch_size"], 8)
        self.assertEqual(record["input_tokens_per_request"], 500)
        self.assertEqual(record["attention_implementation"], "sdpa")
        self.assertEqual(record["model_dimensions"], {
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "vocab_size": 16,
        })
        model.config.head_dim = 32
        explicit = workload_metadata(settings, model, tokenizer, "formatted", inputs, "cuda:0", torch.float16)
        self.assertEqual(explicit["model_dimensions"]["head_dim"], 32)
        token_hash = record["input_token_ids_sha256"]
        assert isinstance(token_hash, str)
        self.assertEqual(len(token_hash), 64)

    def test_output_path_precedence_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "explicit"
            environ = {"DUNLIN_RESULTS": "relative-runs"}
            self.assertEqual(resolve_output_root(output_dir, environ), output_dir.resolve())
            self.assertEqual(resolve_output_root(None, environ), Path("relative-runs").resolve())
            legacy = {"INFERENCE_LAB_RESULTS": "legacy-runs"}
            self.assertEqual(resolve_output_root(None, legacy), Path("legacy-runs").resolve())
            both = {**legacy, **environ}
            self.assertEqual(resolve_output_root(None, both), Path("relative-runs").resolve())
            self.assertEqual(resolve_output_root(output_dir, both), output_dir.resolve())
            self.assertEqual(resolve_output_root(None, {}), (ROOT / "runs").resolve())
            self.assertEqual(
                resolve_output_root(None, {**legacy, "DUNLIN_RESULTS": ""}),
                Path("legacy-runs").resolve(),
            )
            self.assertEqual(
                resolve_config(None, {}, output_dir=output_dir, environ=environ).output_root,
                output_dir.resolve(),
            )

    def test_resolve_revisions_preset_and_resolved(self) -> None:
        """Verify that explicit revisions are preserved and unpinned revisions resolve via AutoConfig."""
        with patch("transformers.AutoConfig.from_pretrained") as mock_autoconfig:
            self.assertEqual(
                resolve_revisions("Example/Model", "a" * 40, "b" * 40),
                ("a" * 40, "b" * 40),
            )
            mock_autoconfig.assert_not_called()

        mock_config = SimpleNamespace(_commit_hash="c" * 40)
        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config) as mock_autoconfig:
            self.assertEqual(
                resolve_revisions("Example/Model"),
                ("c" * 40, "c" * 40),
            )
            mock_autoconfig.assert_called_once_with("Example/Model")

        bad_config = SimpleNamespace(_commit_hash="not-a-40-char-hash")
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=bad_config),
            self.assertRaisesRegex(ValueError, "Could not resolve 40-character commit hash"),
        ):
            resolve_revisions("Example/Model")


if __name__ == "__main__":
    unittest.main()
