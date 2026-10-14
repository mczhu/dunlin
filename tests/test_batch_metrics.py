"""Tests for static-batch construction and reported metrics."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
import sys

import torch
from transformers import BatchEncoding

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from dunlin.workload_helpers import BENCHMARK_PROMPT, DEFAULT_INPUT_TOKENS, repeat_batch
from batch_context_matrix import throughput_speedups as latency_throughput_speedups


class BatchMetricTests(unittest.TestCase):
    def test_prompt_identity_and_default_context_match_recorded_workload(self) -> None:
        self.assertEqual(DEFAULT_INPUT_TOKENS, 500)
        self.assertEqual(
            hashlib.sha256(BENCHMARK_PROMPT.encode()).hexdigest(),
            "9331644019996a4599cc0f38179b836888b5441ef578b9f893999cfd93c11360",
        )

    def test_repeat_batch_preserves_input_values_shapes_and_dtypes(self) -> None:
        inputs = BatchEncoding({
            "input_ids": torch.tensor([[10, 20]], dtype=torch.int64),
            "attention_mask": torch.tensor([[True, False]], dtype=torch.bool),
            "extra": torch.tensor([[[0.25], [0.75]]], dtype=torch.float32),
        })
        originals = {name: value.clone() for name, value in inputs.items()}
        repeated = repeat_batch(inputs, batch_size=3)
        for name, original in originals.items():
            self.assertTrue(torch.equal(inputs[name], original))
            self.assertEqual(repeated[name].dtype, original.dtype)
            self.assertEqual(tuple(repeated[name].shape), (3, *original.shape[1:]))
            for row in repeated[name]:
                self.assertTrue(torch.equal(row, original[0]))

    def test_repeat_batch_repeats_each_model_input(self) -> None:
        inputs = BatchEncoding(
            {
                "input_ids": torch.tensor([[10, 20, 30]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

        repeated = repeat_batch(inputs, batch_size=4)

        self.assertEqual(tuple(repeated["input_ids"].shape), (4, 3))
        self.assertTrue(torch.equal(repeated["input_ids"][3], inputs["input_ids"][0]))
        self.assertTrue(
            torch.equal(repeated["attention_mask"][2], inputs["attention_mask"][0])
        )



    def test_latency_throughput_speedups_include_batch_work(self) -> None:
        self.assertEqual(
            latency_throughput_speedups([10.0, 15.0, 25.0, 40.0]),
            [1.0, 4 / 3, 1.6, 2.0],
        )

    def test_latency_throughput_speedups_rejects_non_finite_and_non_positive_latencies(self) -> None:
        for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -10.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive latency"):
                    latency_throughput_speedups([invalid, 20.0])
                with self.assertRaisesRegex(ValueError, "positive latency"):
                    latency_throughput_speedups([10.0, invalid])


if __name__ == "__main__":
    unittest.main()
