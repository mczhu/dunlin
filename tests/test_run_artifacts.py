"""Check run identity, recorded samples, and summary reconstruction."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from dunlin.artifacts import (
    FORWARD_METRIC_DEFINITIONS,
    FORWARD_METRIC_DEFINITIONS_VERSION,
    create_run_dir,
    new_manifest,
    phase_summary,
    summarize_saved_run,
    write_json,
    write_measurements,
)
from dunlin.environment import collect_environment


class RunArtifactTests(unittest.TestCase):
    def test_single_phase_metrics(self) -> None:
        decode = phase_summary([], [0.01, 0.02, 0.03], 8, 500, ("cached_decode",))
        self.assertEqual(decode["prefill_sample_count"], 0)
        self.assertEqual(decode["median_cached_decode_forward_latency_seconds"], 0.02)
        self.assertEqual(decode["decode_aggregate_tokens_per_second"], 400.0)
        self.assertNotIn("prefill_aggregate_prompt_tokens_per_second", decode)
        prefill = phase_summary([0.1, 0.2, 0.3], [], 8, 500, ("prefill",))
        self.assertEqual(prefill["decode_sample_count"], 0)
        self.assertEqual(prefill["prefill_aggregate_prompt_tokens_per_second"], 20000.0)
        self.assertNotIn("decode_aggregate_tokens_per_second", prefill)
        for phases, prefill_samples, decode_samples in (
            (("cached_decode",), [], []),
            (("cached_decode",), [0.1], [0.01]),
            (("prefill",), [], [0.01]),
            (("unknown",), [], []),
        ):
            with self.subTest(phases=phases), self.assertRaises(ValueError):
                phase_summary(prefill_samples, decode_samples, 8, 500, phases)

    def test_round_trip_samples_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = create_run_dir("example", Path(temporary) / "run")
            manifest = new_manifest("batched_forward_phases", run_dir.name)
            manifest["status"] = "complete"
            write_json(run_dir / "manifest.json", manifest)
            write_json(run_dir / "config.resolved.json", {
                "batch_size": 2,
                "input_tokens_per_request": 10,
                "measured_runs_per_phase": 3,
                "metric_definitions_version": FORWARD_METRIC_DEFINITIONS_VERSION,
                "metric_definitions": FORWARD_METRIC_DEFINITIONS,
            })
            write_measurements(
                run_dir / "measurements.csv", run_dir.name, 2, 10,
                [0.1, 0.2, 0.3], [0.01, 0.02, 0.03],
                [10, 10, 10], [11, 11, 11],
            )
            with (run_dir / "measurements.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[3]["phase"], "cached_decode")
            self.assertEqual(rows[3]["prefill_cache_length"], "10")
            self.assertEqual(rows[3]["decode_cache_length"], "11")
            self.assertEqual(rows[3]["latency_seconds"], "0.01")

            summary = summarize_saved_run(run_dir)
            self.assertNotIn("metric_definitions", summary)
            self.assertEqual(summary["metric_definitions_version"], FORWARD_METRIC_DEFINITIONS_VERSION)
            self.assertEqual(summary, phase_summary(
                [0.1, 0.2, 0.3], [0.01, 0.02, 0.03], 2, 10,
            ))
            self.assertAlmostEqual(summary["prefill_aggregate_prompt_tokens_per_second"], 100.0)
            self.assertAlmostEqual(summary["decode_aggregate_tokens_per_second"], 100.0)

            manifest["status"] = "failed"
            write_json(run_dir / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "completed benchmark"):
                summarize_saved_run(run_dir)
            manifest["status"] = "complete"
            manifest["run_kind"] = "profile"
            write_json(run_dir / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "completed benchmark"):
                summarize_saved_run(run_dir)

    def test_requested_directory_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            create_run_dir("example", path)
            with self.assertRaises(FileExistsError):
                create_run_dir("example", path)

    def test_missing_git_and_gpu_tools_do_not_block_cpu_metadata(self) -> None:
        with patch("dunlin.environment._git", return_value=None):
            environment = collect_environment("cpu", torch.float32)
        self.assertIsNone(environment["git_commit"])
        self.assertIsNone(environment["git_dirty"])
        self.assertEqual(environment["backend"], "cpu")
        self.assertEqual(environment["schema_version"], 2)
        self.assertNotIn("nvidia_driver_version", environment)
        self.assertNotIn("dedicated_device_memory_bytes", environment)
        self.assertNotIn("shared_system_memory_bytes", environment)
        self.assertNotIn("attention_implementation", environment)

    def test_failed_cuda_driver_probe_has_an_explicit_reason(self) -> None:
        properties = SimpleNamespace(name="Example GPU", total_memory=8 * 2**30)
        with (
            patch("dunlin.environment._git", return_value=None),
            patch("dunlin.environment._system_memory_bytes", return_value=16 * 2**30),
            patch("dunlin.environment.torch.cuda.get_device_properties", return_value=properties),
            patch("dunlin.environment.subprocess.run", side_effect=FileNotFoundError),
        ):
            environment = collect_environment("cuda:0", torch.float16)
        self.assertEqual(environment["device_name"], "Example GPU")
        self.assertEqual(environment["dedicated_device_memory_bytes"], 8 * 2**30)
        self.assertIsNone(environment["nvidia_driver_version"])
        reasons = environment["unavailable_fields"]
        assert isinstance(reasons, dict)
        self.assertEqual(
            reasons["nvidia_driver_version"],
            "nvidia-smi query unavailable",
        )

    def test_environment_captures_effective_threads_and_precision_controls(self) -> None:
        """Verify runtime_settings records active PyTorch thread counts and precision controls."""
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            environment = collect_environment("cpu", torch.float32)
            runtime = environment["runtime_settings"]
            assert isinstance(runtime, dict)
            self.assertEqual(runtime["torch_num_threads"], 1)
            self.assertEqual(runtime["torch_num_interop_threads"], torch.get_num_interop_threads())
            self.assertEqual(runtime["float32_matmul_precision"], torch.get_float32_matmul_precision())
            self.assertIn("cuda_matmul_allow_tf32", runtime)
            self.assertIn("cudnn_allow_tf32", runtime)
            self.assertIn("cuda_matmul_allow_fp16_reduced_precision_reduction", runtime)
            self.assertIn("cuda_matmul_allow_bf16_reduced_precision_reduction", runtime)
        finally:
            torch.set_num_threads(original_threads)

    def test_phase_summary_rejects_non_finite_and_non_positive_latencies(self) -> None:
        valid_prefill = [0.1, 0.2]
        valid_decode = [0.01, 0.02]
        for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -0.05):
            with self.subTest(invalid=invalid, phase="prefill"):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    phase_summary([invalid, 0.2], valid_decode, batch_size=1, input_tokens=10)
            with self.subTest(invalid=invalid, phase="decode"):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    phase_summary(valid_prefill, [invalid, 0.02], batch_size=1, input_tokens=10)

    def test_write_measurements_rejects_non_finite_and_non_positive_latencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "measurements.csv"
            for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -0.05):
                with self.subTest(invalid=invalid, phase="prefill"):
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        write_measurements(
                            csv_path, "run-1", 1, 10,
                            [invalid], [0.01], [10], [11],
                        )
                with self.subTest(invalid=invalid, phase="decode"):
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        write_measurements(
                            csv_path, "run-1", 1, 10,
                            [0.1], [invalid], [10], [11],
                        )

    def test_summarize_saved_run_rejects_corrupted_non_finite_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = create_run_dir("example", Path(temporary) / "run")
            manifest = new_manifest("batched_forward_phases", run_dir.name)
            manifest["status"] = "complete"
            write_json(run_dir / "manifest.json", manifest)
            write_json(run_dir / "config.resolved.json", {
                "batch_size": 1,
                "input_tokens_per_request": 10,
                "measured_runs_per_phase": 1,
                "metric_definitions_version": FORWARD_METRIC_DEFINITIONS_VERSION,
                "metric_definitions": FORWARD_METRIC_DEFINITIONS,
            })
            for invalid_str in ("NaN", "nan", "Infinity", "inf", "-Infinity", "-inf", "0.0", "-0.01"):
                with self.subTest(invalid_str=invalid_str):
                    csv_content = (
                        "run_id,phase,repetition_index,batch_size,input_tokens_per_request,"
                        "output_tokens_per_request,prefill_cache_length,decode_cache_length,latency_seconds\n"
                        f"{run_dir.name},prefill,0,1,10,0,,,0.1\n"
                        f"{run_dir.name},cached_decode,0,1,10,1,10,11,{invalid_str}\n"
                    )
                    (run_dir / "measurements.csv").write_text(csv_content)
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        summarize_saved_run(run_dir)


if __name__ == "__main__":
    unittest.main()
