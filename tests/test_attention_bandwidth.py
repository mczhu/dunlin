from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dunlin.attention_bandwidth import (
    METRICS, analyze_session, compare_batches, read_ncu_csv,
)

from dunlin.attention_capture import COUNTER_METRICS, write_capture_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import analyze_attention_bandwidth
import plot_attention_bandwidth


class CounterMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "capture.csv"
        self.units = {metric: "%" for metric in METRICS.values()}
        self.units.update({"gpu__time_duration.sum": "ns", "dram__bytes_read.sum": "byte",
                           "dram__bytes_write.sum": "byte"})
        self.record = {metric: "25" for metric in METRICS.values()}
        self.record.update({"ID": "0", "Kernel Name": "example kernel", "Grid Size": "(2, 1, 224)",
                            "profiler__replayer_passes": "8", "gpu__time_duration.sum": "400,000",
                            "dram__bytes_read.sum": "90,000,000", "dram__bytes_write.sum": "30,000,000"})

    def write_capture(self, records: list[dict[str, str]] | None = None) -> None:
        fields = ["ID", "Kernel Name", "Grid Size", "profiler__replayer_passes", *METRICS.values()]
        with self.path.open("w", newline="") as handle:
            handle.write("==PROF== saved counter export\n")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(self.units)
            writer.writerows(records if records is not None else [self.record])

    def test_decimal_bandwidth_and_explicit_duration_units(self) -> None:
        # 120 MB / 0.4 ms = 300 decimal GB/s, independent of export duration unit.
        for unit, value in (("ns", "400000"), ("us", "400"), ("µs", "400"),
                            ("ms", "0.4"), ("s", "0.0004")):
            with self.subTest(unit=unit):
                self.units["gpu__time_duration.sum"] = unit
                self.record["gpu__time_duration.sum"] = value
                self.write_capture()
                result = read_ncu_csv(self.path)
                self.assertEqual(result["duration_ns"], 400_000)
                self.assertEqual(result["dram_total_bytes"], 120_000_000)
                self.assertEqual(result["dram_gb_per_second"], 300)
                self.assertEqual(result["replay_passes"], 8)

    def test_rejects_multiple_launches_even_with_same_id(self) -> None:
        self.write_capture([self.record, self.record])
        with self.assertRaisesRegex(ValueError, "one launch"):
            read_ncu_csv(self.path)

    def test_rejects_missing_metric_column(self) -> None:
        self.write_capture()
        with self.path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        index = rows[1].index("dram__bytes_read.sum")
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(rows[0])
            writer.writerows([row[:index] + row[index + 1:] for row in rows[1:]])
        with self.assertRaisesRegex(ValueError, "Missing metrics"):
            read_ncu_csv(self.path)

    def test_rejects_missing_units_and_unsupported_units(self) -> None:
        for metric, unit in (("dram__bytes_read.sum", "MB"), ("gpu__time_duration.sum", "cycle"),
                             ("lts__t_sector_hit_rate.pct", "")):
            with self.subTest(metric=metric):
                previous = self.units[metric]
                self.units[metric] = unit
                self.write_capture()
                with self.assertRaises(ValueError):
                    read_ncu_csv(self.path)
                self.units[metric] = previous

    def test_rejects_invalid_counters(self) -> None:
        for metric, value in (("dram__bytes_read.sum", "nan"), ("dram__bytes_write.sum", "-1"),
                              ("gpu__time_duration.sum", "0"), ("dram__bytes_write.sum", "N/A"),
                              ("profiler__replayer_passes", "2.5")):
            with self.subTest(metric=metric, value=value):
                previous = self.record[metric]
                self.record[metric] = value
                self.write_capture()
                with self.assertRaises(ValueError):
                    read_ncu_csv(self.path)
                self.record[metric] = previous

    def test_throughput_rollups_above_100_are_preserved(self) -> None:
        self.record["dram__throughput.avg.pct_of_peak_sustained_elapsed"] = "101.2"
        self.write_capture()
        self.assertEqual(read_ncu_csv(self.path)["dram_pct"], 101.2)

    def test_batch_ratios_use_total_read_and_write_traffic(self) -> None:
        self.write_capture()
        lower = {**read_ncu_csv(self.path), "kernel": "qk", "batch_size": 16}
        upper = {**lower, "batch_size": 32, "duration_ns": 800_000,
                 "dram_total_bytes": 240_000_000, "dram_gb_per_second": 300}
        comparison = compare_batches(lower, upper)
        self.assertEqual(comparison["duration_ratio"], 2)
        self.assertEqual(comparison["dram_bytes_ratio"], 2)
        self.assertEqual(comparison["dram_bandwidth_ratio"], 1)
        with self.assertRaises(ValueError):
            compare_batches(lower, {**upper, "kernel": "av"})
        with self.assertRaises(ValueError):
            compare_batches({**lower, "dram_total_bytes": 0}, upper)

    def test_failed_and_preflight_sessions_are_not_bandwidth_results(self) -> None:
        session = self.path.parent
        (session / "status.txt").write_text("failed (exit 130)\n")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            analyze_session(session)

        (session / "status.txt").write_text("complete\n")
        (session / "settings.txt").write_text("batch=both\nkernel=all\npreflight=True\n")
        with self.assertRaisesRegex(ValueError, "without preflight"):
            analyze_session(session)

    def test_debug_session_is_inspectable_but_rejected_for_official_analysis(self) -> None:
        session = self.make_completed_session(debug=True)
        result = analyze_session(session)
        self.assertEqual(result["source_mode"], "debug")
        self.assertFalse(result["publication_eligible"])
        with self.assertRaisesRegex(ValueError, "Debug sessions"):
            analyze_session(session, require_strict=True)

    def make_completed_session(self, *, debug: bool = False) -> Path:
        session = self.path.parent
        if debug:
            (session / "source-mode.txt").write_text("debug\n")
        for name in ("invocation.sh", "commands.sh", "git-status.txt", "ncu-version.txt",
                     "gpu-before.txt", "gpu-after.txt"):
            (session / name).write_text("")
        (session / "uv.lock").write_text("fixture lockfile\n")
        (session / "status.txt").write_text("complete\n")
        (session / "settings.txt").write_text("batch=both\nkernel=all\npreflight=False\n")
        (session / "started-at.txt").write_text("2026-10-11T15:00:00Z\n")
        (session / "finished-at.txt").write_text("2026-10-11T15:30:00Z\n")
        kernels = ("qk", "kv_expand", "scale", "av", "kv_convert")
        selectors = {kernel: ("example kernel", 0, {16: [2, 1, 224], 32: [2, 1, 224]})
                     for kernel in kernels}
        commit = "a" * 40
        (session / "git-commit.txt").write_text(commit + "\n")
        write_capture_config(session, {
            "schema_version": 1, "kind": "attention_bandwidth_capture",
            "source_mode": "debug" if debug else "strict", "record_origin": "capture_startup",
            "preflight": False,
            "capture": {"device": "0", "nvtx_include": "cached_decode_forward/", "range_filter": "::2",
                        "kernel_name_base": "demangled", "rename_kernels": "off", "launch_count": 1,
                        "replay_mode": "application", "cache_control": "none", "clock_control": "none",
                        "sections": ["LaunchStats"], "metrics": list(COUNTER_METRICS)},
            "workload": {"runner": "experiments/profile_forward_phases.py", "python": "/recorded/python",
                         "phase": "cached_decode", "torch_profiler_enabled": False,
                         "baseline_runs": {str(batch): f"/baseline_b{batch}" for batch in (16, 32)}},
            "points": [{"kernel": kernel, "batch_size": batch, "selector_regex": selectors[kernel][0],
                        "launch_skip": selectors[kernel][1], "expected_grid": selectors[kernel][2][batch]}
                       for kernel in kernels for batch in (16, 32)],
        })
        environment = {"git_commit": commit, "git_dirty": False,
                       "uv_lock_sha256": hashlib.sha256((session / "uv.lock").read_bytes()).hexdigest()}
        self.record["profiler__replayer_passes"] = "1"
        for kernel in kernels:
            for batch in (16, 32):
                stem = f"{kernel}_b{batch}"
                self.path = session / f"{stem}.csv"
                self.write_capture()
                (session / f"{stem}.log").write_text("")
                workload = session / f"{stem}_workloads" / "replay1"
                workload.mkdir(parents=True)
                selection = {"selection_validated": True, "batch_size": batch,
                             "kernel_name": "example kernel", "kernel_id": "0", "grid": [2, 1, 224],
                             "validated_workload_runs": [f"/original/host/{stem}_workloads/replay1"]}
                config = {"batch_size": batch, "phase": "cached_decode", "input_tokens_per_request": 500,
                          "torch_profiler_enabled": False, "baseline_run_dir": f"/baseline_b{batch}",
                          "source_config": f"/baseline_b{batch}/config.source.toml", "output_root": stem,
                          "model_id": "fixture_model", "model_revision": "revision", "tokenizer_revision": "revision",
                          "input_token_ids_sha256": "input", "resolved_dtype": "float16", "resolved_device": "cuda:0",
                          "attention_implementation": "sdpa", "decode_cache_policy": "fresh",
                          "warmup_forwards": 5, "validation_rtol": 0.01, "validation_atol": 0.01}
                manifest = {"status": "complete", "experiment": "profile_forward_phases", "failure": None,
                            "run_id": "replay1", "linked_baseline_run_id": f"baseline_b{batch}",
                            "source_mode": "debug" if debug else "strict"}
                validation = {"forward_matches_reference": True, "logits_finite": True,
                              "logits_shape": [batch, 1, 151936], "cache_length_before": 500,
                              "cache_length_after": 501}
                (session / f"{stem}.selection.json").write_text(json.dumps(selection))
                for name, value in (("manifest", manifest), ("validation", validation),
                                    ("config.resolved", config), ("environment", environment)):
                    (workload / f"{name}.json").write_text(json.dumps(value))
        return session

    def test_debug_source_drift_is_retained_without_loading_committed_selectors(self) -> None:
        session = self.make_completed_session(debug=True)
        (session / "git-status.txt").write_text(" M experiments/profile_attention_bandwidth.py\n")
        (session / "git-commit.txt").write_text("0" * 40 + "\n")
        path = session / "av_b32_workloads/replay1/environment.json"
        environment = json.loads(path.read_text())
        environment.update(git_commit="changed", git_dirty=True, uv_lock_sha256="changed")
        path.write_text(json.dumps(environment))
        with patch("subprocess.run", side_effect=AssertionError("Analysis must not call Git")):
            result = analyze_session(session)
        self.assertEqual(result["capture_provenance"]["source_mode"], "debug")
        capture = next(c for c in result["captures"] if c["kernel"] == "av" and c["batch_size"] == 32)
        self.assertEqual(capture["replay_checks"][0]["environment"], environment)

    def test_debug_analysis_exports_and_plot_remain_labeled(self) -> None:
        session = self.make_completed_session(debug=True)
        output = session / "analysis"
        with patch.object(sys, "argv", ["analyze_attention_bandwidth", str(session), "--output-dir", str(output)]), \
                patch("sys.stdout", new_callable=io.StringIO) as console:
            analyze_attention_bandwidth.main()
        self.assertIn("DEBUG", console.getvalue())
        for name in ("measurements.csv", "comparisons.csv"):
            with (output / name).open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertTrue(rows)
            self.assertTrue(all(row["source_mode"] == "debug" and row["publication_eligible"] == "False"
                                for row in rows))
        plot = output / "plot.png"
        with patch.object(sys, "argv", ["plot_attention_bandwidth", str(output / "summary.json"),
                                       "--output", str(plot)]), \
                patch("matplotlib.figure.Figure.suptitle", autospec=True) as title, \
                patch("sys.stdout", new_callable=io.StringIO):
            plot_attention_bandwidth.main()
        self.assertTrue(title.call_args.args[1].startswith("DEBUG"))
        self.assertTrue(plot.is_file())
        self.assertFalse(json.loads(plot.with_suffix(".sources.json").read_text())["publication_eligible"])

    def test_debug_still_rejects_invalid_evidence(self) -> None:
        session = self.make_completed_session(debug=True)
        for relative, key, value, message in (
            ("av_b32_workloads/replay1/validation.json", "logits_finite", False, "validation failed"),
            ("av_b32_workloads/replay1/config.resolved.json", "model_revision", "different", "Incompatible"),
            ("qk_b16.selection.json", "grid", [999, 1, 1], "selection mismatch"),
            ("qk_b16.selection.json", "validated_workload_runs", [], "Replay records disagree"),
        ):
            with self.subTest(relative=relative, key=key):
                path = session / relative
                original = path.read_text()
                record = json.loads(original)
                record[key] = value
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, message):
                    analyze_session(session)
                path.write_text(original)

    def test_relocated_session_validates_replays_and_retains_input_hashes(self) -> None:
        session = self.make_completed_session()
        result = analyze_session(session)
        self.assertEqual(len(result["captures"]), 10)
        self.assertEqual(len(result["comparisons"]), 5)
        self.assertEqual(result["session_elapsed_seconds"], 1800)
        self.assertEqual(result["captures"][0]["sample_count"], 1)
        self.assertEqual(result["source_sha256"]["qk_b16.csv"],
                         hashlib.sha256((session / "qk_b16.csv").read_bytes()).hexdigest())
        self.assertNotIn("capture-script.py", result["source_sha256"])
        self.assertEqual(result["capture_provenance"]["git_commit"],
                         (session / "git-commit.txt").read_text().strip())

    def test_analysis_needs_no_git_or_capture_script(self) -> None:
        session = self.make_completed_session()
        with patch("subprocess.run", side_effect=AssertionError("Analysis must not call Git")):
            result = analyze_session(session, require_strict=True)
        self.assertEqual(len(result["captures"]), 10)
        self.assertEqual(len(result["comparisons"]), 5)
        self.assertIn("capture-config.json", result["source_sha256"])
        self.assertEqual(result["captures"][0]["selector_regex"], "example kernel")
        self.assertEqual(result["capture_provenance"]["configuration_origin"], "capture_startup")

    def test_missing_or_invalid_config_fails_without_source_fallback(self) -> None:
        session = self.make_completed_session()
        path = session / "capture-config.json"
        original = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing capture-config.json"):
            analyze_session(session)
        for mutation in (
            lambda c: c.update(schema_version=2),
            lambda c: c.update(source_mode="debug"),
            lambda c: c["points"].append(c["points"][0]),
            lambda c: c["points"][0].update(selector_regex="["),
            lambda c: c["points"][0].update(expected_grid=[True, 1, 1]),
            lambda c: c["capture"].update(launch_count=2),
            lambda c: c["capture"].update(metrics=[]),
            lambda c: c["points"].reverse(),
            lambda c: c["workload"]["baseline_runs"].update({"16": "/different_baseline"}),
        ):
            config = json.loads(original)
            mutation(config)
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                analyze_session(session)
        path.write_text(original)

    def test_recorded_expectations_reject_consistent_but_wrong_observations(self) -> None:
        session = self.make_completed_session()
        self.record["Kernel Name"] = "unrelated kernel"
        self.path = session / "qk_b16.csv"
        self.write_capture()
        path = session / "qk_b16.selection.json"
        selection = json.loads(path.read_text())
        selection["kernel_name"] = "unrelated kernel"
        path.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, "Kernel selection mismatch"):
            analyze_session(session, require_strict=True)

    def test_strict_source_provenance_still_rejects_wrong_commit(self) -> None:
        session = self.make_completed_session()
        path = session / "qk_b16_workloads/replay1/environment.json"
        environment = json.loads(path.read_text())
        environment["git_commit"] = "b" * 40
        path.write_text(json.dumps(environment))
        with self.assertRaisesRegex(ValueError, "Unexpected replay configuration/source"):
            analyze_session(session, require_strict=True)

    def test_invalid_capture_commit_has_clear_error(self) -> None:
        session = self.make_completed_session()
        (session / "git-commit.txt").write_text("not-a-commit\n")
        with self.assertRaisesRegex(ValueError, "full capture commit hash"):
            analyze_session(session)

    def test_dirty_capture_is_rejected_before_commit_lookup(self) -> None:
        session = self.make_completed_session()
        (session / "git-status.txt").write_text(" M experiments/profile_attention_bandwidth.py\n")
        (session / "git-commit.txt").write_text("unavailable\n")
        with self.assertRaisesRegex(ValueError, "Capture checkout was dirty"):
            analyze_session(session)

    def test_rejects_failed_correctness_and_changed_workload(self) -> None:
        session = self.make_completed_session()
        workload = session / "av_b32_workloads/replay1"
        for filename, key, value, message in (("validation.json", "logits_finite", False, "validation failed"),
                                              ("config.resolved.json", "model_revision", "different", "Incompatible")):
            with self.subTest(filename=filename):
                path = workload / filename
                original = path.read_text()
                record = json.loads(original)
                record[key] = value
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, message):
                    analyze_session(session)
                path.write_text(original)

    def test_rejects_wrong_selector_and_missing_replay_record(self) -> None:
        session = self.make_completed_session()
        path = session / "qk_b16.selection.json"
        original = path.read_text()
        record = json.loads(original)
        record["grid"] = [999, 1, 1]
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "selection mismatch"):
            analyze_session(session)
        path.write_text(original)
        record = json.loads(original)
        record["validated_workload_runs"] = []
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Replay records disagree"):
            analyze_session(session)


if __name__ == "__main__":
    unittest.main()
