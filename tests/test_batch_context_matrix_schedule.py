"""Check that a fresh context-by-batch run measures every requested shape."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from batch_context_matrix import BATCH_SIZES, CONTEXT_LENGTHS, collect_measurements, run_benchmark


class BatchContextMatrixScheduleTests(unittest.TestCase):
    def test_child_receives_requested_device_and_dtype(self) -> None:
        summary = {
            "median_prefill_forward_latency_seconds": 0.01,
            "median_cached_decode_forward_latency_seconds": 0.02,
        }
        with (
            patch("batch_context_matrix.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="")) as run,
            patch("batch_context_matrix.summarize_saved_run", return_value=summary),
        ):
            self.assertEqual(
                run_benchmark(32, 2, Path("/tmp/child"), "cuda:1", "float16"),
                (10.0, 20.0),
            )
        command = run.call_args.args[0]
        self.assertEqual(command[-4:], ["--device", "cuda:1", "--dtype", "float16"])

    def test_child_receives_revisions_from_settings(self) -> None:
        """Verify that run_benchmark forwards model and tokenizer revisions to child CLI commands."""
        summary = {
            "median_prefill_forward_latency_seconds": 0.01,
            "median_cached_decode_forward_latency_seconds": 0.02,
        }
        from dunlin.config import resolve_config

        settings = resolve_config(
            None,
            {
                "context_lengths": [32],
                "batch_sizes": [1],
                "model_revision": "a" * 40,
                "tokenizer_revision": "b" * 40,
            },
        )
        with (
            patch("batch_context_matrix.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="")) as run,
            patch("batch_context_matrix.summarize_saved_run", return_value=summary),
        ):
            self.assertEqual(
                run_benchmark(32, 1, Path("/tmp/child"), settings=settings),
                (10.0, 20.0),
            )
        command = run.call_args.args[0]
        self.assertIn("--model-revision", command)
        self.assertEqual(command[command.index("--model-revision") + 1], "a" * 40)
        self.assertIn("--tokenizer-revision", command)
        self.assertEqual(command[command.index("--tokenizer-revision") + 1], "b" * 40)

    def test_main_resolves_revisions_and_records_in_parent(self) -> None:
        """Verify that matrix main() resolves unpinned revisions and records them in parent config."""
        import json
        import tempfile
        from batch_context_matrix import main

        resolved_hash = "f" * 40
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary)
            test_args = [
                "batch_context_matrix.py",
                "--output-dir", str(out_root),
                "--context-lengths", "32",
                "--batch-sizes", "1",
            ]
            import io
            from contextlib import redirect_stdout

            output = io.StringIO()
            with (
                patch("sys.argv", test_args),
                patch("batch_context_matrix.SourceCheckout"),
                patch("batch_context_matrix.resolve_revisions", return_value=(resolved_hash, resolved_hash)) as mock_resolve,
                patch("batch_context_matrix.run_matrix") as mock_run_matrix,
                redirect_stdout(output),
            ):
                main()
            mock_resolve.assert_called_once()
            mock_run_matrix.assert_called_once()

            run_dir = mock_run_matrix.call_args.args[0]
            config_json = json.loads((run_dir / "config.resolved.json").read_text())
            self.assertEqual(config_json["model_revision"], resolved_hash)
            self.assertEqual(config_json["tokenizer_revision"], resolved_hash)

    def test_every_configured_point_is_measured_once(self) -> None:
        called: list[tuple[int, int]] = []

        def benchmark(context_length: int, batch_size: int) -> tuple[float, float]:
            called.append((context_length, batch_size))
            return float(context_length), float(batch_size)

        measurements = collect_measurements(benchmark)
        expected = [
            (context_length, batch_size)
            for context_length in CONTEXT_LENGTHS
            for batch_size in BATCH_SIZES
        ]

        self.assertEqual(called, expected)
        self.assertEqual(len(called), len(CONTEXT_LENGTHS) * len(BATCH_SIZES))
        self.assertEqual(
            measurements,
            {
                context_length: {
                    batch_size: (float(context_length), float(batch_size))
                    for batch_size in BATCH_SIZES
                }
                for context_length in CONTEXT_LENGTHS
            },
        )


if __name__ == "__main__":
    unittest.main()
