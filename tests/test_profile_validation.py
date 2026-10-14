"""Validate capture evidence and failure handling without a GPU or checkpoint."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from dunlin.config import resolve_config
from profile_forward_phases import CapturedForward, main, validate_profile_forward


class ProfileValidationTests(unittest.TestCase):
    def captured(self) -> CapturedForward:
        return CapturedForward(torch.ones((2, 1, 4)), 500, 501, "DynamicCache", 3)

    def validate(self, captured: CapturedForward, **overrides) -> dict[str, object]:
        settings = {
            "phase": "cached_decode", "batch_size": 2, "input_tokens": 500,
            "vocab_size": 4, "use_torch_profiler": True,
            "reference_logits": torch.ones((2, 1, 4)),
            **overrides,
        }
        return validate_profile_forward(captured, **settings)

    def test_decode_report_records_observed_values(self) -> None:
        record = self.validate(self.captured())
        self.assertEqual(record, {
            "logits_shape": [2, 1, 4], "logits_finite": True,
            "profiled_matches_unprofiled": True, "forward_matches_reference": True,
            "cache_length_before": 500, "cache_length_after": 501,
            "cache_class": "DynamicCache", "cuda_kernel_events": 3,
        })

    def test_prefill_and_nvtx_semantics(self) -> None:
        prefill = replace(self.captured(), cache_length_before=None,
                          cache_length_after=500, cache_class=None)
        self.assertEqual(self.validate(prefill, phase="prefill")["cache_length_after"], 500)
        nvtx = replace(self.captured(), cuda_kernel_events=None)
        record = self.validate(nvtx, use_torch_profiler=False)
        self.assertIsNone(record["cuda_kernel_events"])
        self.assertIsNone(record["profiled_matches_unprofiled"])
        self.assertTrue(record["forward_matches_reference"])

    def test_invalid_capture_rejected(self) -> None:
        for changes, message in (
            ({"cuda_kernel_events": None}, "no events"),
            ({"cuda_kernel_events": 0}, "no CUDA kernel events"),
            ({"cache_length_after": None}, "Expected past_key_values"),
            ({"cache_length_before": 501}, "fresh prompt cache length"),
            ({"cache_length_before": None}, "fresh prompt cache length"),
            ({"cache_length_after": 502}, "Expected cache length 501"),
            ({"logits": torch.ones((2, 2, 4))}, "Invalid profiled logits"),
            ({"logits": torch.full((2, 1, 4), float("nan"))}, "Invalid profiled logits"),
            ({"logits": torch.full((2, 1, 4), float("inf"))}, "Invalid profiled logits"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, message):
                self.validate(replace(self.captured(), **changes))

    def test_reference_shape_and_tolerance(self) -> None:
        captured = self.captured()
        # A broadcastable reference must not accidentally pass output agreement.
        with self.assertRaisesRegex(RuntimeError, "Invalid reference logits"):
            self.validate(captured, reference_logits=torch.ones((1, 1, 4)))
        near = replace(captured, logits=captured.logits + 0.001)
        self.assertTrue(self.validate(near)["forward_matches_reference"])
        with self.assertRaisesRegex(RuntimeError, "differ from an unprofiled"):
            self.validate(near, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "differ from an unprofiled"):
            self.validate(captured, reference_logits=torch.zeros((2, 1, 4)))

    def test_runner_writes_validation_only_after_success(self) -> None:
        for matching_reference in (True, False):
            with self.subTest(matching_reference=matching_reference), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "profile"
                settings = resolve_config(None, {
                    "context_lengths": [500], "batch_sizes": [2],
                    "warmup_runs_per_phase": 1,
                })
                model = Mock()
                model.config = SimpleNamespace(
                    hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, vocab_size=4,
                )
                inputs = {"input_ids": torch.zeros((2, 500), dtype=torch.long)}
                reference = torch.ones((2, 1, 4)) if matching_reference else torch.zeros((2, 1, 4))
                # Disposable output is ignored; only the selected capture is validated.
                disposable = replace(self.captured(), cache_length_after=502, cuda_kernel_events=0)
                selected = self.captured()
                with (
                    patch("sys.argv", ["profile", "--batch-size", "2", "--run-dir", str(run_dir)]),
                    patch("profile_forward_phases.SourceCheckout"),
                    patch("profile_forward_phases.resolve_config", return_value=settings),
                    patch("profile_forward_phases.select_device", return_value=("cuda:0", torch.float32)),
                    patch("profile_forward_phases.load_workload", return_value=(model, None, "", {}, inputs)),
                    patch("profile_forward_phases.workload_metadata", return_value={}),
                    patch("profile_forward_phases.collect_environment", return_value={}),
                    patch("profile_forward_phases.prepare_decode_inputs", return_value=(None, None, None)),
                    patch("profile_forward_phases.synchronize"),
                    patch("profile_forward_phases.capture_forward", side_effect=[disposable, selected]),
                    patch("profile_forward_phases.reference_forward", return_value=reference),
                    patch("profile_forward_phases.validate_profile_forward", wraps=validate_profile_forward) as validate,
                    redirect_stdout(io.StringIO()),
                ):
                    if matching_reference:
                        main()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "differ from an unprofiled"):
                            main()
                validate.assert_called_once()
                self.assertIs(validate.call_args.args[0], selected)
                manifest = json.loads((run_dir / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete" if matching_reference else "failed")
                self.assertEqual((run_dir / "validation.json").exists(), matching_reference)
                if matching_reference:
                    record = json.loads((run_dir / "validation.json").read_text())
                    self.assertEqual(record["cache_length_after"], 501)
                    self.assertTrue(record["forward_matches_reference"])
                else:
                    self.assertIn("differ from an unprofiled", manifest["failure"])


if __name__ == "__main__":
    unittest.main()
