"""Check asynchronous attribution, elapsed-time accounting, and FLOP estimates."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from dunlin.trace_attention import (
    analyze_attention_events, decode_flops_per_layer_per_request, interval_union_us,
)


def event(name: str, category: str, start: float, duration: float,
          external_id: int | None = None, tid: int = 1) -> dict:
    return {
        "ph": "X", "name": name, "cat": category, "pid": 1, "tid": tid,
        "ts": start, "dur": duration,
        "args": {} if external_id is None else {
            "External id": external_id, "device": 0, "context": 1, "stream": 7,
        },
    }


def example_trace() -> list[dict]:
    return [
        event("cached_decode_forward", "user_annotation", 100, 50),
        event("wait_for_completion", "user_annotation", 150, 70),
        event("aten::scaled_dot_product_attention", "cpu_op", 110, 10, 1),
        event("aten::bmm", "cpu_op", 112, 2, 2),
        event("aten::copy_", "cpu_op", 116, 2, 3),
        event("aten::mm", "cpu_op", 140, 2, 4),
        # GPU execution follows the host forward: containment cannot attribute it.
        event("attention product", "kernel", 160, 10, 2),
        event("attention copy", "kernel", 170, 20, 3),
        event("projection", "kernel", 200, 10, 4),
        # These annotations/flows must not inflate device activity.
        event("cached_decode_forward", "gpu_user_annotation", 160, 50),
        {"ph": "f", "cat": "ac2g", "ts": 160, "id": 1000},
        event("preparation", "kernel", 20, 10, 999),
    ]


class TraceAttentionTests(unittest.TestCase):
    def test_union_counts_nested_overlapping_and_adjacent_intervals_once(self) -> None:
        self.assertEqual(interval_union_us([(8, 12), (1, 5), (2, 3), (4, 8)]), 11)
        self.assertEqual(interval_union_us([]), 0)
        for interval in [(2, 1), (0, float("nan")), (0, float("inf"))]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                interval_union_us([interval])

    def test_attribution_uses_cpu_ancestry_and_ids_not_gpu_temporal_overlap(self) -> None:
        metrics = analyze_attention_events(example_trace())
        self.assertAlmostEqual(metrics["completion_ms"], 0.120)
        self.assertAlmostEqual(metrics["device_activity_ms"], 0.040)
        self.assertAlmostEqual(metrics["attention_activity_ms"], 0.030)
        self.assertAlmostEqual(metrics["outside_sdpa_activity_ms"], 0.010)
        self.assertAlmostEqual(metrics["attention_share_of_device_activity"], 0.75)
        self.assertAlmostEqual(metrics["forward_activity_gap_ms"], 0.080)
        self.assertAlmostEqual(metrics["attention_internal_gap_sum_ms"], 0)
        self.assertEqual(metrics["selected_device_event_count"], 3)
        self.assertEqual(metrics["ignored_device_event_count"], 1)

    def test_overlapping_groups_are_reported_without_inflating_total_activity(self) -> None:
        trace = example_trace()
        trace[7]["ts"] = 165
        metrics = analyze_attention_events(trace)
        self.assertAlmostEqual(metrics["device_activity_ms"], 0.035)
        self.assertAlmostEqual(metrics["group_overlap_excess_ms"], 0.005)
        self.assertAlmostEqual(metrics["device_event_overlap_excess_ms"], 0.005)
        self.assertAlmostEqual(metrics["attention_activity_ms"], 0.025)
        self.assertAlmostEqual(sum(metrics["attention_direct_owner_duration_sums_ms"].values()), 0.030)

    def test_attention_gaps_are_separate_from_kernel_execution(self) -> None:
        trace = example_trace()
        trace[7]["ts"] = 175
        metrics = analyze_attention_events(trace)
        self.assertAlmostEqual(metrics["attention_internal_gap_sum_ms"], 0.005)
        self.assertAlmostEqual(metrics["attention_activity_ms"], 0.030)

    def test_kernel_ranking_splits_geometry_scope_and_ancestry(self) -> None:
        trace = example_trace()
        trace += [event("aten::clone", "cpu_op", 115, 4, 8),
                  event("aten::copy_", "cpu_op", 113, 0.5, 9),
                  event("attention copy", "kernel", 191, 3, 9),
                  event("attention copy", "kernel", 195, 4, 9),
                  event("attention copy", "kernel", 211, 2, 4)]
        trace[7]["args"].update(grid=[8, 1, 1], block=[128, 1, 1])
        for e in trace[-3:-1]:
            e["args"].update(grid=[4, 1, 1], block=[128, 1, 1])
        ranking = analyze_attention_events(trace)["kernel_ranking"]
        self.assertEqual(len(ranking["rows"]), 5)
        largest = ranking["rows"][0]
        self.assertEqual(largest["grid"], [8, 1, 1])
        self.assertEqual(largest["cpu_ancestry_names"],
                         ["aten::scaled_dot_product_attention", "aten::clone", "aten::copy_"])
        self.assertEqual(largest["first_launch"]["external_id"], 3)
        repeated = next(r for r in ranking["rows"] if r["grid"] == [4, 1, 1])
        self.assertEqual(repeated["launch_count"], 2)
        self.assertEqual(repeated["mean_us"], 3.5)
        self.assertEqual(repeated["median_us"], 3.5)
        self.assertAlmostEqual(repeated["duration_sum_ms"], 0.007)
        self.assertAlmostEqual(repeated["share_of_sdpa_kernel_duration_sum"], 7 / 37)
        self.assertEqual(len(ranking["first_sdpa_launches"]), 4)
        self.assertEqual(ranking["first_sdpa_launches"][0]["forward_kernel_ordinal"], 0)

    def test_kernel_ranking_distinguishes_sum_union_and_non_kernel_events(self) -> None:
        trace = example_trace() + [event("attention copy", "kernel", 180, 15, 3),
                                   event("transfer", "gpu_memcpy", 195, 2, 3)]
        ranking = analyze_attention_events(trace)["kernel_ranking"]
        largest = ranking["rows"][0]
        self.assertAlmostEqual(largest["duration_sum_ms"], 0.035)
        self.assertAlmostEqual(largest["duration_union_ms"], 0.025)
        self.assertAlmostEqual(ranking["sdpa_kernel_duration_sum_ms"], 0.045)
        self.assertEqual(len(ranking["first_sdpa_launches"]), 3)

    def test_first_sdpa_is_chronological_with_shuffled_trace_entries(self) -> None:
        trace = example_trace() + [event("aten::scaled_dot_product_attention", "cpu_op", 125, 5, 8),
                                   event("later attention", "kernel", 193, 2, 8)]
        metrics = analyze_attention_events(list(reversed(trace)))
        self.assertEqual([e["kernel_name"] for e in metrics["kernel_ranking"]["first_sdpa_launches"]],
                         ["attention product", "attention copy"])
        self.assertAlmostEqual(metrics["attention_activity_ms"], 0.032)

    def test_zero_kernel_duration_has_no_defined_kernel_share(self) -> None:
        trace = example_trace()
        for e in trace:
            if e.get("cat") == "kernel":
                e["dur"] = 0
        trace.append(event("transfer", "gpu_memcpy", 180, 2, 3))
        ranking = analyze_attention_events(trace)["kernel_ranking"]
        self.assertEqual(ranking["forward_kernel_duration_sum_ms"], 0)
        self.assertTrue(all(r["share_of_forward_kernel_duration_sum"] is None for r in ranking["rows"]))

    def test_unknown_device_work_inside_window_is_rejected(self) -> None:
        trace = example_trace() + [event("unknown", "gpu_memcpy", 190, 2, 999)]
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            analyze_attention_events(trace)

    def test_foreign_host_thread_is_not_a_child_of_sdpa(self) -> None:
        trace = example_trace() + [event("aten::mul", "cpu_op", 112, 2, 8, tid=2),
                                   event("foreign", "kernel", 195, 2, 8)]
        with self.assertRaisesRegex(ValueError, "Unattributed"):
            analyze_attention_events(trace)

    def test_incomplete_or_ambiguous_capture_is_rejected(self) -> None:
        cases = []
        cases.append(example_trace()[1:])  # Missing forward.
        cases.append(example_trace() + [example_trace()[0]])  # Duplicate forward.
        duplicate_id = example_trace()
        duplicate_id[5]["args"]["External id"] = 2
        cases.append(duplicate_id)
        nested_sdpa = example_trace() + [event("aten::scaled_dot_product_attention", "cpu_op", 111, 8, 8)]
        cases.append(nested_sdpa)
        late_device = example_trace()
        late_device[8]["ts"] = 215
        cases.append(late_device)
        for trace in cases:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                analyze_attention_events(trace)

    def test_decode_arithmetic_counts_use_query_heads_and_available_cache(self) -> None:
        config = {
            "phase": "cached_decode", "decode_tokens_per_request": 1,
            "input_tokens_per_request": 500,
            "architecture": {"hidden_size": 896, "intermediate_size": 4864,
                             "num_attention_heads": 14, "num_key_value_heads": 2,
                             "head_dim": 64},
        }
        self.assertEqual(decode_flops_per_layer_per_request(config), {
            "attention_products": 1795584, "qkvo_projections": 3670016,
            "mlp_projections": 26148864,
        })
        config["decode_tokens_per_request"] = 2
        with self.assertRaises(ValueError):
            decode_flops_per_layer_per_request(config)


class StandalonePlotValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        for attr, filename in (("execution", "plot_trace_attention.py"), ("dram", "plot_attention_bandwidth.py")):
            spec = importlib.util.spec_from_file_location(attr, root / "experiments" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            setattr(cls, attr, module)

    def setUp(self) -> None:
        self.trace = {"runs": [
            {"config": {"phase": "cached_decode"}, "environment": {"git_dirty": False},
             "manifest": {"status": "complete"},
             "validation": {"profiled_matches_unprofiled": True, "forward_matches_reference": True,
                            "logits_finite": True}}
        ]}
        replay = {"manifest": {"status": "complete"},
                  "validation": {"forward_matches_reference": True, "logits_finite": True}}
        self.bandwidth = {
            "environment": {"git_dirty": False}, "configs_by_batch": {"16": {}, "32": {}},
            "captures": [{"kernel": kernel, "batch_size": batch, "sample_count": 1, "replay_passes": 8,
                          "dram_pct": 90.0, "replay_checks": [deepcopy(replay) for _ in range(8)]}
                         for kernel in ("qk", "av", "kv_expand", "scale", "kv_convert")
                         for batch in (16, 32)],
        }

    def test_each_plot_validates_its_own_source_independently(self) -> None:
        self.assertFalse(self.execution.validate_execution_source(self.trace, require_strict=True))
        self.assertFalse(self.dram.validate_dram_source(self.bandwidth, require_strict=True))
        self.trace["runs"][0]["environment"]["git_dirty"] = True
        with self.assertRaisesRegex(ValueError, "clean capture"):
            self.execution.validate_execution_source(self.trace, require_strict=True)
        self.bandwidth["environment"]["git_dirty"] = True
        with self.assertRaisesRegex(ValueError, "clean capture"):
            self.dram.validate_dram_source(self.bandwidth, require_strict=True)

    def test_duplicate_missing_and_nonfinite_counter_samples_fail(self) -> None:
        duplicate = deepcopy(self.bandwidth)
        duplicate["captures"][-1] = duplicate["captures"][0]
        missing = deepcopy(self.bandwidth)
        missing["captures"].pop()
        nonfinite = deepcopy(self.bandwidth)
        nonfinite["captures"][0]["dram_pct"] = float("nan")
        missing_replay = deepcopy(self.bandwidth)
        missing_replay["captures"][0]["replay_checks"].pop()
        for changed in (duplicate, missing, nonfinite, missing_replay):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.dram.validate_dram_source(changed, require_strict=False)

    def test_clean_debug_inputs_remain_ineligible(self) -> None:
        for source, summary, validate in (
            (self.trace["runs"][0]["manifest"], self.trace, self.execution.validate_execution_source),
            (self.bandwidth, self.bandwidth, self.dram.validate_dram_source),
            (self.bandwidth["captures"][0]["replay_checks"][0]["manifest"], self.bandwidth,
             self.dram.validate_dram_source),
        ):
            source["source_mode"] = "debug"
            self.assertTrue(validate(summary, require_strict=False))
            with self.assertRaisesRegex(ValueError, "rejects debug"):
                validate(summary, require_strict=True)
            source.pop("source_mode")

    def test_failed_correctness_is_rejected_even_for_debug(self) -> None:
        self.trace["source_mode"] = "debug"
        self.trace["runs"][0]["validation"]["forward_matches_reference"] = False
        with self.assertRaisesRegex(ValueError, "correctness"):
            self.execution.validate_execution_source(self.trace, require_strict=False)
        self.bandwidth["source_mode"] = "debug"
        self.bandwidth["captures"][0]["replay_checks"][0]["validation"]["logits_finite"] = False
        with self.assertRaisesRegex(ValueError, "correctness"):
            self.dram.validate_dram_source(self.bandwidth, require_strict=False)


if __name__ == "__main__":
    unittest.main()
