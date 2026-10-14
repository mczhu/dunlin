"""Check baseline-derived profiler settings without loading a model or GPU."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from profile_forward_phases import (  # noqa: E402
    BASELINE_WORKLOAD_KEYS,
    capture_forward,
    settings_from_baseline,
    validate_baseline_workload,
)


class ProfileBaselineTests(unittest.TestCase):
    def baseline(self) -> dict[str, object]:
        record: dict[str, object] = {key: "same" for key in BASELINE_WORKLOAD_KEYS}
        record.update({
            "experiment": "batched_forward_phases",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "prompt_recipe_id": "city_planning_qwen_chat_v1",
            "batch_size": 8,
            "input_tokens_per_request": 500,
            "decode_tokens_per_request": 1,
            "resolved_device": "cuda:0",
            "resolved_dtype": "torch.float16",
            "warmup_runs_per_phase": 3,
            "measured_runs_per_phase": 30,
            "use_cache": True,
            "logits_to_keep": 1,
            "compiled": False,
        })
        return record

    def test_derives_effective_workload_and_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            baseline = self.baseline()
            (run_dir / "manifest.json").write_text(json.dumps({
                "run_id": "n500_b8", "status": "complete", "run_kind": "benchmark",
                "experiment": "batched_forward_phases",
            }))
            (run_dir / "config.resolved.json").write_text(json.dumps(baseline))
            settings, run_id, loaded = settings_from_baseline(run_dir, run_dir / "profiles")
            self.assertEqual(run_id, "n500_b8")
            self.assertEqual(settings.context_lengths, (500,))
            self.assertEqual(settings.batch_sizes, (8,))
            self.assertEqual(settings.device, "cuda:0")
            self.assertEqual(settings.dtype, "float16")
            self.assertEqual(settings.warmup_runs_per_phase, 3)
            self.assertEqual(settings.model_revision, "a" * 40)
            validate_baseline_workload(loaded, baseline)
            with self.assertRaisesRegex(ValueError, "input_token_ids_sha256"):
                validate_baseline_workload(loaded, {**baseline, "input_token_ids_sha256": "different"})

    def test_rejects_nonbaseline_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "manifest.json").write_text(json.dumps({
                "run_id": "n500_b8", "status": "failed", "run_kind": "benchmark",
                "experiment": "batched_forward_phases",
            }))
            (run_dir / "config.resolved.json").write_text(json.dumps(self.baseline()))
            with self.assertRaisesRegex(ValueError, "completed batched_forward_phases"):
                settings_from_baseline(run_dir, None)

    def test_nvtx_forward_skips_torch_profiler(self) -> None:
        inputs = {"input_ids": torch.zeros((1, 2), dtype=torch.long)}
        fresh_cache = SimpleNamespace(get_seq_length=lambda: 2)
        output_cache = SimpleNamespace(get_seq_length=lambda: 3)
        calls = []

        def model(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=torch.ones((1, 1, 4)), past_key_values=output_cache)

        with (patch("profile_forward_phases.prepare_decode_inputs",
                    return_value=(torch.ones((1, 1), dtype=torch.long), None, fresh_cache)),
              patch("profile_forward_phases.synchronize") as sync,
              patch("profile_forward_phases.profile", side_effect=AssertionError("Torch profiler started")),
              patch("profile_forward_phases.torch.cuda.nvtx.range", return_value=nullcontext()) as nvtx):
            captured = capture_forward(
                model, inputs, "cuda:0", "cached_decode", save_to=None,
                record_shapes=False, use_torch_profiler=False,
            )

        nvtx.assert_called_once_with("cached_decode_forward")
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(captured.logits.shape), (1, 1, 4))
        self.assertEqual(captured.cache_length_before, 2)
        self.assertEqual(captured.cache_length_after, 3)
        self.assertEqual(captured.cache_class, "SimpleNamespace")
        self.assertIsNone(captured.cuda_kernel_events)


if __name__ == "__main__":
    unittest.main()
