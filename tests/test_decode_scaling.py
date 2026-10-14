"""Check scaling metrics and rejection of unmatched or incomplete evidence."""

from __future__ import annotations

import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from dunlin.artifacts import (
    FORWARD_METRIC_DEFINITIONS, FORWARD_METRIC_DEFINITIONS_VERSION,
    new_manifest, write_json, write_measurements,
)
from dunlin.decode_scaling import analyze_decode_runs, latency_spread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import analyze_decode_scaling
import plot_decode_scaling


class DecodeScalingTests(unittest.TestCase):
    def make_run(self, root: Path, batch: int, samples: list[float]) -> Path:
        path = root / f"b{batch}"
        path.mkdir()
        manifest = new_manifest("batched_forward_phases", path.name)
        manifest["status"] = "complete"
        config = {
            "experiment": "batched_forward_phases", "batch_size": batch,
            "input_tokens_per_request": 500, "decode_tokens_per_request": 1,
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct", "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40, "model_class": "Qwen2ForCausalLM",
            "tokenizer_class": "Qwen2Tokenizer", "prompt_recipe_id": "city_planning_qwen_chat_v1",
            "prompt_text_sha256": "c" * 64, "formatted_prompt_sha256": "d" * 64,
            "input_token_ids_sha256": "e" * 64, "resolved_dtype": "torch.float32",
            "attention_implementation": "sdpa", "compiled": False,
            "use_cache": True, "logits_to_keep": 1,
            "decode_cache_policy": "fresh prompt prefill before each timed decode forward",
            "timing_scope": "synchronized model forward", "warmup_runs_per_phase": 5,
            "measured_runs_per_phase": len(samples), "phases": ["cached_decode"],
            "metric_definitions_version": FORWARD_METRIC_DEFINITIONS_VERSION,
            "metric_definitions": FORWARD_METRIC_DEFINITIONS,
        }
        environment = {
            "git_commit": "f" * 40, "git_dirty": False, "uv_lock_sha256": "0" * 64,
            "python_version": "3.14", "torch_version": "2.13", "transformers_version": "5.16",
            "backend": "cpu", "device": "cpu", "dtype": "torch.float32",
            "device_name": "test CPU", "platform": "test Linux", "architecture": "x86_64",
            "os_release": "test", "wsl": False, "system_physical_memory_bytes": 1024,
            "runtime_settings": {"torch_num_threads": 2},
        }
        write_json(path / "manifest.json", manifest)
        write_json(path / "config.resolved.json", config)
        write_json(path / "environment.json", environment)
        write_measurements(path / "measurements.csv", path.name, batch, 500,
                           [], samples, [500] * len(samples), [501] * len(samples))
        # Deliberately incorrect saved medians must not affect reconstruction.
        write_json(path / "summary.json", {"decode_aggregate_tokens_per_second": -1})
        return path

    def change(self, path: Path, filename: str, key: str, value: object) -> None:
        record = json.loads((path / filename).read_text())
        record[key] = value
        write_json(path / filename, record)

    def test_reverse_order_preserved_and_metrics_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lower = self.make_run(root, 8, [0.010, 0.020, 0.030])
            upper = self.make_run(root, 16, [0.020, 0.040, 0.060])
            result = analyze_decode_runs([upper, lower])
            self.assertEqual(result["source_run_ids_in_supplied_order"], ["b16", "b8"])
            self.assertEqual([point["batch_size"] for point in result["points"]], [8, 16])
            self.assertEqual(result["points"][0]["tokens_per_second"], 400)
            self.assertEqual(result["points"][0]["sample_count"], 3)
            interval = result["adjacent_doublings"][0]
            self.assertEqual(interval["throughput_gain_fraction"], 0)
            self.assertTrue(interval["weakened_scaling_candidate"])
            self.assertTrue(interval["below_plateau_threshold"])
            self.assertNotIn("plateau_established", result)

    def test_linear_gain_and_threshold_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [self.make_run(root, batch, [latency] * 3)
                    for batch, latency in ((8, 0.020), (16, 0.020), (32, 0.032))]
            intervals = analyze_decode_runs(runs)["adjacent_doublings"]
            self.assertEqual(intervals[0]["throughput_gain_fraction"], 1)
            self.assertFalse(intervals[0]["weakened_scaling_candidate"])
            self.assertAlmostEqual(intervals[1]["throughput_gain_fraction"], 0.25)
            self.assertFalse(intervals[1]["weakened_scaling_candidate"])
            self.assertTrue(intervals[1]["threshold_sensitivity"]["0.5"])

    def test_sample_spread_is_inclusive(self) -> None:
        spread = latency_spread([1, 2, 3, 4, 5])
        self.assertEqual(spread["median_seconds"], 3)
        self.assertAlmostEqual(spread["p10_seconds"], 1.4)
        self.assertAlmostEqual(spread["p90_seconds"], 4.6)

    def test_host_memory_tolerance(self) -> None:
        for difference in (4096, 1024 * 1024, 1024 * 1024 + 1):
            with self.subTest(difference=difference), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lower = self.make_run(root, 16, [0.02] * 3)
                upper = self.make_run(root, 32, [0.03] * 3)
                memory = 16550289408
                self.change(lower, "environment.json", "system_physical_memory_bytes", memory)
                self.change(upper, "environment.json", "system_physical_memory_bytes", memory - difference)
                for runs in ([lower, upper], [upper, lower]):
                    if difference > 1024 * 1024:
                        with self.assertRaisesRegex(ValueError, "system_physical_memory_bytes"):
                            analyze_decode_runs(runs)
                    else:
                        result = analyze_decode_runs(runs)
                        expected = memory if runs[0] == lower else memory - difference
                        self.assertEqual(result["environment"]["system_physical_memory_bytes"], expected)

    def test_incompatible_conditions_rejected(self) -> None:
        for filename, key, value, expected in (
            ("config.resolved.json", "input_token_ids_sha256", "changed", "input_token_ids_sha256"),
            ("config.resolved.json", "phases", ["prefill", "cached_decode"], "Incomplete measurement"),
            ("environment.json", "device_name", "other CPU", "device_name"),
            ("environment.json", "runtime_settings", {"torch_num_threads": 4}, "runtime_settings"),
            ("environment.json", "git_commit", "other", "git_commit"),
            ("environment.json", "git_dirty", True, "clean source"),
            ("environment.json", "uv_lock_sha256", None, "missing environment provenance"),
            ("manifest.json", "status", "failed", "completed benchmark"),
            ("manifest.json", "source_mode", "debug", "debug runs"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lower = self.make_run(root, 8, [0.02] * 3)
                upper = self.make_run(root, 16, [0.03] * 3)
                self.change(upper, filename, key, value)
                with self.assertRaisesRegex(ValueError, expected):
                    analyze_decode_runs([lower, upper], require_strict=True)

    def test_debug_source_drift_is_inspectable_but_not_official_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lower = self.make_run(root, 8, [0.02] * 3)
            upper = self.make_run(root, 16, [0.03] * 3)
            self.change(upper, "manifest.json", "source_mode", "debug")
            # Even a clean debug capture is development evidence.
            with self.assertRaisesRegex(ValueError, "debug runs"):
                analyze_decode_runs([lower, upper], require_strict=True)
            self.change(upper, "environment.json", "git_dirty", True)
            self.change(upper, "environment.json", "git_commit", "changed")
            self.change(upper, "environment.json", "uv_lock_sha256", "changed")
            result = analyze_decode_runs([lower, upper])
            self.assertEqual(result["source_mode"], "debug")
            self.assertFalse(result["publication_eligible"])
            self.assertEqual(result["source_provenance"][1]["git_commit"], "changed")
            self.assertTrue(result["source_provenance"][1]["git_dirty"])
            self.assertAlmostEqual(result["points"][1]["tokens_per_second"], 16 / 0.03)
            self.change(upper, "config.resolved.json", "input_token_ids_sha256", "changed")
            with self.assertRaisesRegex(ValueError, "input_token_ids_sha256"):
                analyze_decode_runs([lower, upper])
            self.change(upper, "config.resolved.json", "input_token_ids_sha256", "e" * 64)
            samples_file = upper / "measurements.csv"
            samples_file.write_text(samples_file.read_text().replace(",0.03", ",nan", 1))
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                analyze_decode_runs([lower, upper])

    def test_invalid_samples_and_missing_doubling_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lower = self.make_run(root, 8, [0.02] * 3)
            upper = self.make_run(root, 32, [0.03] * 3)
            with self.assertRaisesRegex(ValueError, "double batch"):
                analyze_decode_runs([lower, upper])
            with self.assertRaisesRegex(ValueError, "unique"):
                analyze_decode_runs([lower, lower])
            samples_file = lower / "measurements.csv"
            samples_file.write_text(samples_file.read_text().replace(",0.02", ",nan", 1))
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                analyze_decode_runs([lower])

    def test_debug_analysis_and_plot_cli_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, 8, [0.02, 0.03, 0.04])
            self.change(run, "manifest.json", "source_mode", "debug")
            self.change(run, "environment.json", "git_dirty", True)
            output = root / "analysis"
            with patch.object(sys, "argv", ["analyze_decode_scaling", str(run), "--output-dir", str(output)]), \
                    patch("sys.stdout", new_callable=io.StringIO) as console:
                analyze_decode_scaling.main()
            self.assertIn("DEBUG", console.getvalue())
            self.assertFalse(json.loads((output / "summary.json").read_text())["publication_eligible"])
            plot = output / "plot.png"
            with patch.object(sys, "argv", ["plot_decode_scaling", str(output / "summary.json"),
                                           "--output", str(plot)]), \
                    patch("matplotlib.figure.Figure.suptitle", autospec=True) as title, \
                    patch("sys.stdout", new_callable=io.StringIO):
                plot_decode_scaling.main()
            self.assertTrue(title.call_args.args[1].startswith("DEBUG"))
            self.assertTrue(plot.is_file())
            self.assertFalse(json.loads(plot.with_suffix(".sources.json").read_text())["publication_eligible"])
            rejected_output = root / "official"
            with patch.object(sys, "argv", ["analyze_decode_scaling", str(run), "--output-dir",
                                           str(rejected_output), "--require-strict"]), \
                    patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                analyze_decode_scaling.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(rejected_output.exists())


if __name__ == "__main__":
    unittest.main()
