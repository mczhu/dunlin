"""Check matrix postprocessing from saved samples and comparison provenance."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from dunlin.artifacts import (
    FORWARD_METRIC_DEFINITIONS,
    FORWARD_METRIC_DEFINITIONS_VERSION,
    new_manifest,
    write_json,
    write_measurements,
)
from dunlin.matrix_results import (
    MATRIX_METRIC_DEFINITIONS,
    compare_matrix_runs,
    load_matrix_run,
)
import plot_batch_context_matrix


def saved_matrix(root: Path, name: str, *, backend: str = "cpu") -> Path:
    run_dir = root / name
    run_dir.mkdir()
    manifest = new_manifest("batch_context_matrix", name)
    manifest["status"] = "complete"
    manifest["matrix_points"] = [
        {"context_length": 32, "batch_size": batch, "run_id": f"n32_b{batch}", "status": "complete"}
        for batch in (1, 2)
    ]
    write_json(run_dir / "manifest.json", manifest)
    parent_config = {
        "model_id": "Example/Qwen2", "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40, "prompt_recipe_id": "test-recipe",
        "context_lengths": [32], "batch_sizes": [1, 2], "resolved_dtype": "torch.float16",
        "resolved_device": backend, "warmup_runs_per_phase": 5,
        "measured_runs_per_phase": 2, "phases": ["prefill", "cached_decode"],
        "decode_tokens_per_request": 1, "use_cache": True, "logits_to_keep": 1,
        "point_process_policy": "one model process per point in context-then-batch order",
        "metric_definitions_version": "batch_context_matrix_metrics_v1",
        "metric_definitions": MATRIX_METRIC_DEFINITIONS,
    }
    write_json(run_dir / "config.resolved.json", parent_config)
    environment = {
        "backend": backend, "device": backend, "device_name": backend,
        "platform": backend, "architecture": "example",
        "git_commit": "c" * 40, "git_dirty": False,
        "uv_lock_sha256": "d" * 64, "python_version": "3.13",
        "torch_version": "2.13", "transformers_version": "5.16",
    }
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "summary.json", {"matrix_points": ["stale"]})
    for batch in (1, 2):
        child_id = f"n32_b{batch}"
        child_dir = run_dir / child_id
        child_dir.mkdir()
        child_manifest = new_manifest("batched_forward_phases", child_id)
        child_manifest["status"] = "complete"
        write_json(child_dir / "manifest.json", child_manifest)
        write_json(child_dir / "environment.json", environment)
        write_json(child_dir / "config.resolved.json", {
            **parent_config,
            "experiment": "batched_forward_phases", "model_class": "Qwen2ForCausalLM",
            "tokenizer_class": "Qwen2Tokenizer", "batch_size": batch,
            "input_tokens_per_request": 32, "prompt_text_sha256": "e" * 64,
            "formatted_prompt_sha256": "f" * 64,
            "input_token_ids_sha256": "1" * 64,
            "attention_implementation": "sdpa", "compiled": False,
            "decode_cache_policy": "fresh prompt prefill before each timed decode forward",
            "timing_scope": "synchronized model forwards",
            "metric_definitions_version": FORWARD_METRIC_DEFINITIONS_VERSION,
            "metric_definitions": FORWARD_METRIC_DEFINITIONS,
        })
        prefill = [0.1, 0.3] if batch == 1 else [0.2, 0.4]
        decode = [0.01, 0.03] if batch == 1 else [0.02, 0.04]
        write_measurements(
            child_dir / "measurements.csv", child_id, batch, 32,
            prefill, decode, [32, 32], [33, 33],
        )
    return run_dir


class MatrixResultsTests(unittest.TestCase):
    def test_debug_matrices_are_inspectable_but_not_official_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_matrix_run(saved_matrix(root, "baseline"))
            debug_dir = saved_matrix(root, "debug")
            for manifest_path in debug_dir.rglob("manifest.json"):
                manifest = json.loads(manifest_path.read_text())
                manifest["source_mode"] = "debug"
                write_json(manifest_path, manifest)
            debug_run = load_matrix_run(debug_dir)
            with self.assertRaisesRegex(ValueError, "Debug runs"):
                compare_matrix_runs(baseline, debug_run)

            child_env_path = debug_dir / "n32_b2" / "environment.json"
            environment = json.loads(child_env_path.read_text())
            environment.update(git_dirty=True, git_commit="d" * 40)
            write_json(child_env_path, environment)
            debug_run = load_matrix_run(debug_dir)
            self.assertEqual(len(debug_run.summary["matrix_points"]), 2)
            parent_env_path = debug_dir / "environment.json"
            environment = json.loads(parent_env_path.read_text())
            environment.update(git_dirty=True, git_commit="changed", uv_lock_sha256="changed")
            write_json(parent_env_path, environment)
            comparison = compare_matrix_runs(baseline, load_matrix_run(debug_dir), allow_debug=True)
            self.assertEqual(comparison["source_mode"], "debug")
            self.assertFalse(comparison["publication_eligible"])
            self.assertEqual(comparison["source_provenance"][1]["git_commit"], "changed")

            plot = root / "comparison.png"
            with patch.object(sys, "argv", ["plot_batch_context_matrix", str(baseline.path),
                                           "--compare", str(debug_dir), "--output", str(plot)]), \
                    patch.object(plot_batch_context_matrix, "render_plot") as render, \
                    patch("sys.stdout", new_callable=io.StringIO):
                plot_batch_context_matrix.main()
            self.assertTrue(render.call_args.args[2].startswith("DEBUG"))
            self.assertIn("DEBUG", render.call_args.args[0][1][0])
            sources = json.loads(plot.with_suffix(".sources.json").read_text())
            self.assertFalse(sources["publication_eligible"])
            self.assertEqual(sources["comparison"]["source_provenance"][1]["git_commit"], "changed")

            for child_config_path in debug_dir.glob("n32_b*/config.resolved.json"):
                config = json.loads(child_config_path.read_text())
                config["input_token_ids_sha256"] = "changed"
                write_json(child_config_path, config)
            with self.assertRaisesRegex(ValueError, "input_token_ids_sha256"):
                compare_matrix_runs(baseline, load_matrix_run(debug_dir), allow_debug=True)

            # A strict parent must not silently consume debug children.
            parent_manifest = json.loads((debug_dir / "manifest.json").read_text())
            parent_manifest["source_mode"] = "strict"
            write_json(debug_dir / "manifest.json", parent_manifest)
            with self.assertRaisesRegex(ValueError, "source mode differs"):
                load_matrix_run(debug_dir)

    def test_rebuilds_summary_from_samples_and_rejects_missing_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = saved_matrix(Path(temporary), "baseline")
            run = load_matrix_run(run_dir)
            self.assertEqual(len(run.summary["matrix_points"]), 2)
            point = run.summary["matrix_points"][1]
            self.assertAlmostEqual(point["median_prefill_forward_latency_ms"], 300)
            self.assertAlmostEqual(point["decode_throughput_speedup_vs_batch_one"], 4 / 3)

            child_csv = run_dir / "n32_b2" / "measurements.csv"
            with child_csv.open(newline="") as source:
                rows = list(csv.DictReader(source))
            with child_csv.open("w", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete measurement data"):
                load_matrix_run(run_dir)

            manifest = new_manifest("batch_context_matrix", "baseline")
            manifest["status"] = "complete"
            manifest["matrix_points"] = [
                {"context_length": 32, "batch_size": 1, "run_id": "n32_b1", "status": "complete"}
            ]
            write_json(run_dir / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing, duplicate, or reordered points"):
                load_matrix_run(run_dir)

    def test_rejects_child_with_mismatched_model_or_tokenizer_revision(self) -> None:
        """Verify that matrix validation rejects children whose revisions diverge from parent config."""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = saved_matrix(Path(temporary), "mismatched")
            parent_config_path = run_dir / "config.resolved.json"
            parent_config = json.loads(parent_config_path.read_text())
            parent_config["model_revision"] = None
            write_json(parent_config_path, parent_config)
            with self.assertRaisesRegex(ValueError, "differs from matrix config: model_revision"):
                load_matrix_run(run_dir)

    def test_comparison_requires_matching_inputs_and_explicit_machine_choice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = load_matrix_run(saved_matrix(root, "baseline"))
            candidate_dir = saved_matrix(root, "candidate", backend="cuda:0")
            candidate = load_matrix_run(candidate_dir)
            with self.assertRaisesRegex(ValueError, "allow-cross-machine"):
                compare_matrix_runs(baseline, candidate)
            comparison = compare_matrix_runs(baseline, candidate, allow_cross_machine=True)
            self.assertTrue(comparison["cross_machine"])
            self.assertEqual(comparison["machine_differences"]["backend"]["candidate"], "cuda:0")

            config_path = candidate_dir / "n32_b1" / "config.resolved.json"
            config = json.loads(config_path.read_text())
            config["input_token_ids_sha256"] = "2" * 64
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "Input token IDs differ across batches"):
                load_matrix_run(candidate_dir)
            other_config_path = candidate_dir / "n32_b2" / "config.resolved.json"
            other_config = json.loads(other_config_path.read_text())
            other_config["input_token_ids_sha256"] = "2" * 64
            write_json(other_config_path, other_config)
            with self.assertRaisesRegex(ValueError, "input_token_ids_sha256"):
                compare_matrix_runs(
                    baseline, load_matrix_run(candidate_dir), allow_cross_machine=True,
                )

    def test_comparison_detects_runtime_settings_thread_and_precision_differences(self) -> None:
        """Verify compare_matrix_runs detects different runtime_settings (e.g. CPU thread count or precision)."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = saved_matrix(root, "baseline")
            candidate_dir = saved_matrix(root, "candidate")

            # Update baseline parent and children with 8 threads
            for path in [baseline_dir / "environment.json", baseline_dir / "n32_b1" / "environment.json", baseline_dir / "n32_b2" / "environment.json"]:
                env = json.loads(path.read_text())
                env["runtime_settings"] = {"torch_num_threads": 8, "float32_matmul_precision": "highest"}
                write_json(path, env)

            # Update candidate parent and children with 1 thread
            for path in [candidate_dir / "environment.json", candidate_dir / "n32_b1" / "environment.json", candidate_dir / "n32_b2" / "environment.json"]:
                env = json.loads(path.read_text())
                env["runtime_settings"] = {"torch_num_threads": 1, "float32_matmul_precision": "high"}
                write_json(path, env)

            # Verify that a mismatch within a run is rejected
            child_env_path = candidate_dir / "n32_b1" / "environment.json"
            child_env = json.loads(child_env_path.read_text())
            child_env["runtime_settings"] = {"torch_num_threads": 4, "float32_matmul_precision": "high"}
            write_json(child_env_path, child_env)
            with self.assertRaisesRegex(ValueError, "differs from matrix environment: runtime_settings"):
                load_matrix_run(candidate_dir)
            child_env["runtime_settings"] = {"torch_num_threads": 1, "float32_matmul_precision": "high"}
            write_json(child_env_path, child_env)

            baseline = load_matrix_run(baseline_dir)
            candidate = load_matrix_run(candidate_dir)
            with self.assertRaisesRegex(ValueError, "machine/backend differs"):
                compare_matrix_runs(baseline, candidate)

            comparison = compare_matrix_runs(baseline, candidate, allow_cross_machine=True)
            self.assertTrue(comparison["cross_machine"])
            self.assertIn("runtime_settings", comparison["machine_differences"])
            self.assertEqual(
                comparison["machine_differences"]["runtime_settings"]["baseline"]["torch_num_threads"],
                8,
            )
            self.assertEqual(
                comparison["machine_differences"]["runtime_settings"]["candidate"]["torch_num_threads"],
                1,
            )



if __name__ == "__main__":
    unittest.main()
