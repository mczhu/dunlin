"""Check phase selection, fresh caches, and saved single-phase summaries."""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from batched_forward_phases import main, run_experiment
from dunlin.artifacts import new_manifest, summarize_saved_run, write_json
from dunlin.config import PHASES, resolve_config


class PhaseSelectionTests(unittest.TestCase):
    def test_selected_phases_and_artifact_round_trip(self) -> None:
        for phases in (("prefill",), ("cached_decode",), PHASES):
            with self.subTest(phases=phases), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                settings = resolve_config(None, {
                    "warmup_runs_per_phase": 1, "measured_runs_per_phase": 2,
                }, default_context_lengths=(500,), default_batch_sizes=(8,))
                inputs = {"input_ids": torch.zeros((8, 500), dtype=torch.long)}
                caches = []

                def prepare(*args):
                    cache = SimpleNamespace(length=500)
                    cache.get_seq_length = lambda: cache.length
                    caches.append(cache)
                    return torch.zeros((8, 1), dtype=torch.long), None, cache

                def forward(**kwargs):
                    cache = kwargs.get("past_key_values")
                    if cache is not None:
                        self.assertEqual(cache.length, 500)
                        cache.length += 1
                    return SimpleNamespace(past_key_values=cache)

                model = Mock(side_effect=forward)
                manifest = new_manifest("batched_forward_phases", run_dir.name)
                with (
                    patch("batched_forward_phases.SourceCheckout"),
                    patch("batched_forward_phases.select_device", return_value=("cpu", torch.float32)),
                    patch("batched_forward_phases.load_workload", return_value=(model, None, "", {}, inputs)),
                    patch("batched_forward_phases.workload_metadata", return_value={
                        "batch_size": 8, "input_tokens_per_request": 500,
                    }),
                    patch("batched_forward_phases.collect_environment", return_value={}),
                    patch("batched_forward_phases.prepare_decode_inputs", side_effect=prepare),
                    patch("batched_forward_phases.synchronize"),
                    redirect_stdout(io.StringIO()),
                ):
                    run_experiment(settings, run_dir, manifest, phases)

                self.assertEqual(model.call_count, 3 * len(phases))
                self.assertEqual(len(caches), 3 if "cached_decode" in phases else 0)
                self.assertEqual(len({id(cache) for cache in caches}), len(caches))
                config = json.loads((run_dir / "config.resolved.json").read_text())
                self.assertEqual(config["phases"], list(phases))
                with (run_dir / "measurements.csv").open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 2 * len(phases))
                self.assertEqual({row["phase"] for row in rows}, set(phases))
                manifest["status"] = "complete"
                write_json(run_dir / "manifest.json", manifest)
                summary = json.loads((run_dir / "summary.json").read_text())
                self.assertEqual(summarize_saved_run(run_dir), summary)
                for phase, prefix in (("prefill", "prefill"), ("cached_decode", "decode")):
                    self.assertEqual(summary[f"{prefix}_sample_count"], 2 if phase in phases else 0)
                    self.assertEqual(f"median_{phase}_forward_latency_seconds" in summary, phase in phases)

                # A missing selected-phase sample must still fail reconstruction.
                (run_dir / "measurements.csv").write_text(
                    "\n".join((run_dir / "measurements.csv").read_text().splitlines()[:-1]) + "\n"
                )
                with self.assertRaisesRegex(ValueError, "Incomplete measurement data"):
                    summarize_saved_run(run_dir)

    def test_dry_run_phase_selection_and_default(self) -> None:
        for phase in (None, "prefill", "cached_decode"):
            argv = ["benchmark", "--batch-size", "8", "--input-tokens", "500", "--device", "cpu", "--dry-run"]
            if phase:
                argv += ["--phase", phase]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(output):
                main()
            self.assertEqual(json.loads(output.getvalue())["phases"], list(PHASES) if phase is None else [phase])


if __name__ == "__main__":
    unittest.main()
