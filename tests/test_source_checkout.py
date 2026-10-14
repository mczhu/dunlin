"""Check the shared source policy and experiment entry points without workloads."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from dunlin.source import SourceCheckout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


class SourceCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "experiment.py"
        self.source.write_text("# committed source\n")
        (self.root / ".gitignore").write_text("runs/\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit()
        self.run_dir = self.root / "runs" / "fixture"
        self.run_dir.mkdir(parents=True)

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self):
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                 "commit", "--quiet", "--allow-empty", "-m", "Fixture")

    def start(self):
        return SourceCheckout.start(self.run_dir, repo_root=self.root)

    def records(self):
        return [json.loads(line) for line in
                (self.run_dir / "source-checks.jsonl").read_text().splitlines()]

    def test_clean_checkout_and_ignored_outputs_pass_and_record_commit(self):
        source = self.start()
        (self.run_dir / "output.csv").write_text("latency\n1\n")
        source.check("completion")
        self.assertEqual(source.commit, self.git("rev-parse", "HEAD"))
        self.assertEqual([r["stage"] for r in self.records()], ["startup", "completion"])
        self.assertTrue(all(r["passed"] and not r["git_status"] for r in self.records()))

    def test_staged_unstaged_and_untracked_files_reject_startup(self):
        for mode in ("staged", "unstaged", "untracked"):
            with self.subTest(mode=mode):
                if mode == "untracked":
                    extra = self.root / "new.py"
                    extra.write_text("# new\n")
                else:
                    self.source.write_text("# edited\n")
                    if mode == "staged":
                        self.git("add", "experiment.py")
                with self.assertRaisesRegex(ValueError, "must be clean at startup"):
                    self.start()
                self.assertFalse(self.records()[-1]["passed"])
                self.git("restore", "--staged", "--worktree", "experiment.py")
                if mode == "untracked":
                    extra.unlink()

    def test_edits_during_a_run_fail_the_next_boundary(self):
        source = self.start()
        self.source.write_text("# edited during capture\n")
        with self.assertRaisesRegex(ValueError, "must be clean at after capture"):
            source.check("after capture")
        self.assertFalse(self.records()[-1]["passed"])

    def test_debug_records_dirty_source_and_commit_changes_without_rejecting(self):
        self.source.write_text("# development changes\n")
        source = SourceCheckout.start(self.run_dir, repo_root=self.root, debug=True)
        self.git("add", "experiment.py")
        self.commit()
        source.check("completion")
        records = self.records()
        self.assertTrue(all(r["source_mode"] == "debug" and r["accepted"] for r in records))
        self.assertTrue(all(not r["passed"] for r in records))
        self.assertTrue(records[0]["git_status"])
        self.assertNotEqual(records[1]["git_commit"], records[1]["expected_commit"])

    def test_committing_during_a_run_fails_even_with_a_clean_worktree(self):
        source = self.start()
        self.commit()
        with self.assertRaisesRegex(ValueError, "commit changed at completion"):
            source.check("completion")
        record = self.records()[-1]
        self.assertFalse(record["passed"])
        self.assertNotEqual(record["git_commit"], record["expected_commit"])

    def test_git_unavailable_fails_closed(self):
        with patch("dunlin.source.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "Cannot verify experiment checkout"):
                self.start()

    def test_stdout_only_runs_send_provenance_to_stderr(self):
        output = io.StringIO()
        with redirect_stderr(output):
            source = SourceCheckout.start(repo_root=self.root)
            source.check("completion")
        records = [json.loads(line.removeprefix("source_check: "))
                   for line in output.getvalue().splitlines()]
        self.assertEqual([r["stage"] for r in records], ["startup", "completion"])


    def test_matrix_marks_the_changed_point_failed_and_stops(self):
        import batch_context_matrix as matrix
        from dunlin.config import resolve_config

        source = self.start()
        settings = resolve_config(None, {}, default_context_lengths=(500, 1000),
                                  default_batch_sizes=(1, 2))
        manifest = {"matrix_points": [
            {"context_length": n, "batch_size": b, "run_id": f"n{n}_b{b}", "status": "pending"}
            for n in settings.context_lengths for b in settings.batch_sizes
        ]}

        def changed_point(*args, **kwargs):
            self.source.write_text("# edited during matrix point\n")
            return 1.0, 1.0

        with patch.object(matrix, "run_benchmark", side_effect=changed_point) as benchmark:
            with self.assertRaisesRegex(ValueError, "must be clean at after N=500 B=1"):
                matrix.run_matrix(self.run_dir, manifest, settings, source=source)
            self.assertEqual(benchmark.call_count, 1)
        saved = json.loads((self.run_dir / "manifest.json").read_text())
        self.assertEqual([p["status"] for p in saved["matrix_points"]],
                         ["failed", "pending", "pending", "pending"])

    def test_debug_forward_completes_with_dirty_source_and_saves_mode(self):
        import batched_forward_phases as runner
        from unittest.mock import Mock

        self.source.write_text("# development version\n")
        original_start = SourceCheckout.start
        args = ["forward", "--batch-size", "1", "--input-tokens", "32", "--phase", "prefill",
                "--device", "cpu", "--warmup-runs", "1", "--measured-runs", "2",
                "--output-dir", str(self.run_dir), "--debug"]
        model = Mock()
        with (patch("sys.argv", args),
              patch.object(runner.SourceCheckout, "start", side_effect=lambda run_dir, **kwargs:
                           original_start(run_dir, repo_root=self.root, **kwargs)) as start,
              patch.object(runner, "load_workload", return_value=(model, None, "", {}, {})),
              patch.object(runner, "workload_metadata", return_value={
                  "batch_size": 1, "input_tokens_per_request": 32,
              }), patch.object(runner, "collect_environment", return_value={}),
              redirect_stdout(io.StringIO())):
            runner.main()
        start.assert_called_once()
        self.assertTrue(start.call_args.kwargs["debug"])
        manifest_path, = self.run_dir.glob("*/manifest.json")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["source_mode"], "debug")
        self.assertIn("_debug_", manifest["run_id"])
        self.assertEqual(model.call_count, 3)
        checks = [json.loads(line) for line in
                  (manifest_path.parent / "source-checks.jsonl").read_text().splitlines()]
        self.assertTrue(all(c["source_mode"] == "debug" and c["accepted"] for c in checks))


    def test_debug_flag_reaches_actual_child_commands(self):
        from types import SimpleNamespace

        stdout = "\n".join([
            "median_generation_latency_seconds: 1.0",
            "median_prefill_forward_latency_seconds: 1.0",
            "median_cached_decode_forward_latency_seconds: 1.0",
            "median_batch_wall_latency_seconds: 1.0",
            "median_aggregate_output_tokens_per_second: 1.0",
        ])
        import batch_context_matrix as matrix
        with (patch.object(matrix.subprocess, "run", return_value=SimpleNamespace(
                  stdout=stdout, returncode=0)) as child,
              patch.object(matrix, "summarize_saved_run", return_value={
                  "median_prefill_forward_latency_seconds": 1.0,
                  "median_cached_decode_forward_latency_seconds": 1.0,
              }), redirect_stdout(io.StringIO())):
            matrix.run_benchmark(32, 1, run_dir=self.run_dir, debug=True)
        self.assertIn("--debug", child.call_args.args[0])



class ExperimentStartupTests(unittest.TestCase):
    def test_dry_runs_do_not_check_source_or_create_artifacts(self):
        for name in ("batched_forward_phases", "profile_forward_phases", "batch_context_matrix"):
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as temporary:
                module = importlib.import_module(name)
                args = [name, "--dry-run", "--output-dir", temporary]
                if name != "batch_context_matrix":
                    args += ["--batch-size", "1"]
                with (patch("sys.argv", args),
                      patch.object(module.SourceCheckout, "start") as start,
                      patch.object(module, "select_device", return_value=("cuda:0", torch.float32)),
                      redirect_stdout(io.StringIO())):
                    module.main()
                start.assert_not_called()
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_every_runner_rejects_dirty_source_before_loading_or_launching(self):
        runners = {
            "batched_forward_phases": ["--batch-size", "1"],
            "profile_forward_phases": ["--batch-size", "1"],
            "batch_context_matrix": [],
        }
        for name, args in runners.items():
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as temporary:
                module = importlib.import_module(name)
                if name in ("batched_forward_phases", "profile_forward_phases", "batch_context_matrix"):
                    args = [*args, "--output-dir", temporary]
                with (patch("sys.argv", [name, *args]),
                      patch.object(module.SourceCheckout, "start",
                                   side_effect=ValueError("Experiment checkout must be clean at startup")),
                      redirect_stdout(io.StringIO())):
                    # Device resolution is allowed, model loading must never happen.
                    if hasattr(module, "load_workload"):
                        with (patch.object(module, "select_device", return_value=("cuda:0", torch.float32)),
                              patch.object(module, "load_workload") as load):
                            with self.assertRaisesRegex(ValueError, "must be clean"):
                                module.main()
                            load.assert_not_called()
                    elif name == "batch_context_matrix":
                        with patch.object(module, "resolve_revisions") as resolve:
                            with self.assertRaisesRegex(ValueError, "must be clean"):
                                module.main()
                            resolve.assert_not_called()
                    elif hasattr(module, "run_benchmark"):
                        with patch.object(module, "run_benchmark") as launch:
                            with self.assertRaisesRegex(ValueError, "must be clean"):
                                module.main()
                            launch.assert_not_called()
                    else:
                        with patch.object(module.AutoTokenizer, "from_pretrained") as load:
                            with self.assertRaisesRegex(ValueError, "must be clean"):
                                module.main()
                            load.assert_not_called()
                if name in ("batched_forward_phases", "profile_forward_phases", "batch_context_matrix"):
                    manifests = list(Path(temporary).glob("*/manifest.json"))
                    self.assertEqual(len(manifests), 1)
                    self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
