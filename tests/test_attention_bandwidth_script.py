"""Exercise capture orchestration with fake tools; no CUDA or model execution."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/profile_attention_bandwidth.py"
SPEC = importlib.util.spec_from_file_location("profile_attention_bandwidth", SCRIPT)
bandwidth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bandwidth)

FAKE_NCU = r'''
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
with Path(os.environ['BANDWIDTH_TEST_LOG']).open('a') as log:
    log.write(json.dumps(args) + '\n')
mode = os.environ.get('BANDWIDTH_TEST_MODE', '')
if '--import' in args:
    if '--page' in args and args[args.index('--page') + 1] == 'raw' and '--print-metric-name' in args:
        print("==ERROR== Option '--print-metric-name' is only supported for the details page.")
        raise SystemExit(1)
    report = json.loads(Path(args[args.index('--import') + 1]).read_text())
    if 'internal::gemvx' in report['name'] and ('--rename-kernels' not in args or args[args.index('--rename-kernels') + 1] != 'off'):
        report['name'] = report['name'].replace('internal::gemvx::', '')
    writer = csv.writer(sys.stdout)
    if mode == 'long_csv':
        writer.writerow(['ID', 'Kernel Name', 'Grid Size', 'Metric Name', 'Metric Unit', 'Metric Value'])
        writer.writerow(['0', report['name'], report['grid'], 'gpu__time_duration.sum', 'nsecond', '400000'])
        writer.writerow(['0', report['name'], report['grid'], 'dram__bytes_read.sum', 'byte', '120000000'])
    else:
        metric = 'gpu__time_duration.sum' if report['counters'] else 'launch__grid_size'
        writer.writerow(['ID', 'Kernel Name', 'Grid Size', metric, 'dram__bytes_read.sum'])
        writer.writerow(['', '', '', 'ns', 'byte'])
        if mode != 'empty_csv':
            writer.writerow(['0', report['name'], report['grid'], '400000', '120000000'])
            if mode == 'multiple_kernels':
                writer.writerow(['1', report['name'], report['grid'], '400000', '120000000'])
elif '-o' in args:
    if mode == 'interrupt':
        child_code = """
import os
from pathlib import Path
import signal
import time
root = Path(os.environ['BANDWIDTH_TEST_LOG']).parent
def stop(signum, frame):
    (root / 'child-stopped').write_text('stopped')
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
(root / 'child-ready').write_text('ready')
time.sleep(60)
"""
        child = subprocess.Popen([sys.executable, '-c', child_code])
        def stop(signum, frame):
            child.wait(timeout=5)
            raise SystemExit(130)
        signal.signal(signal.SIGINT, stop)
        time.sleep(60)
    if mode == 'counter_failure':
        print('ERR_NVGPUCTRPERM: test counter access failure', file=sys.stderr)
        raise SystemExit(7)
    if mode == 'dirty_checkout':
        with Path('experiments/profile_attention_bandwidth.py').open('a') as source:
            source.write('\n# edited while collection was running\n')
    if mode == 'changed_commit':
        subprocess.run(['git', '-c', 'user.name=Dunlin Tests', '-c', 'user.email=tests@example.invalid',
                        '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
                        'commit', '--quiet', '--allow-empty', '-m', 'Commit during collection'], check=True)
    if mode == 'changed_config':
        path = Path(args[args.index('-o') + 1]).parent / 'capture-config.json'
        config = json.loads(path.read_text())
        config['points'][0]['launch_skip'] += 1
        path.write_text(json.dumps(config))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    batch = int(args[args.index('--baseline-run-dir') + 1].rsplit('_b', 1)[1])
    pattern = args[args.index('--kernel-name') + 1]
    if 'gemv2T' in pattern:
        name, grid = 'void gemv2T_kernel_val<int>', [63, 1, batch * 14]
    elif 'unrolled_elementwise_kernel' in pattern:
        name, grid = 'void unrolled_elementwise_kernel<direct_copy_kernel_cuda>', [2004 if batch == 16 else 4008, 1, 1]
    elif 'elementwise_kernel' in pattern:
        name, grid = 'void elementwise_kernel<128, 2, direct_copy_kernel_cuda>', [28056 if batch == 16 else 56112, 1, 1]
    elif 'AUnaryFunctor' in pattern:
        name, grid = 'void AUnaryFunctor<float, MulFunctor<float>>', [7014 if batch == 16 else 14028, 1, 1]
    else:
        name, grid = 'void internal::gemvx::kernel<int>', [2, 1, batch * 14]
    if mode == 'wrong_grid':
        grid = [1, 1, 1]
    if mode == 'wrong_name':
        name = 'unrelated_kernel'
    report_path = Path(args[args.index('-o') + 1] + '.ncu-rep')
    report_path.write_text(json.dumps({'name': name, 'grid': str(grid), 'counters': '--metrics' in args or 'SpeedOfLight' in args}))
    if mode == 'missing_report':
        report_path.unlink()
    # Simulate multiple application replays with independent runner artifacts.
    workload_root = Path(args[args.index('--output-dir') + 1])
    for replay in range(2):
        workload = workload_root / f'pass_{replay}'
        workload.mkdir(parents=True)
        (workload / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
        (workload / 'environment.json').write_text(json.dumps({
            'git_commit': commit,
            'git_dirty': dirty or (mode == 'dirty_replay' and replay == 0),
        }))
        (workload / 'validation.json').write_text(json.dumps({
            'forward_matches_reference': mode != 'invalid_outputs' or replay == 0,
            'logits_finite': True, 'logits_shape': [batch, 1, 151936],
            'cache_length_before': 500, 'cache_length_after': 501,
        }))
else:
    if mode == 'dirty_setup' and '--query-metrics' in args:
        with Path('experiments/profile_attention_bandwidth.py').open('a') as source:
            source.write('\n# edited during profiler setup\n')
    print('fake ncu metadata')
'''


class AttentionBandwidthScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "checkout"
        self.capture_script = self.checkout / "experiments/profile_attention_bandwidth.py"
        self.capture_script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.capture_script)
        (self.checkout / "uv.lock").write_text("# fixture lockfile\n")
        (self.checkout / ".gitignore").write_text("runs/\n")
        subprocess.run(["git", "init", "--quiet", str(self.checkout)], check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=self.checkout, check=True, capture_output=True)
        self.commit_checkout("Initial capture fixture")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        ncu = self.bin / "ncu"
        ncu.write_text(f"#!{sys.executable}\n" + FAKE_NCU)
        ncu.chmod(0o755)
        gpu = self.bin / "nvidia-smi"
        gpu.write_text("#!/bin/sh\nprintf 'fake GPU metadata\\n'\n")
        gpu.chmod(0o755)
        self.prefix = self.root / "baseline prefix"
        for batch in (16, 32):
            directory = Path(f"{self.prefix}_b{batch}")
            directory.mkdir()
            for filename in ("manifest.json", "config.resolved.json"):
                (directory / filename).write_text("{}")
        self.output = self.root / "output with spaces"
        self.log = self.root / "ncu-calls.jsonl"
        self.env = {**os.environ, "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                    "BANDWIDTH_TEST_LOG": str(self.log)}

    def commit_checkout(self, message):
        subprocess.run(["git", "-c", "user.name=Dunlin Tests", "-c", "user.email=tests@example.invalid",
                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                        "commit", "--quiet", "-m", message],
                       cwd=self.checkout, check=True, capture_output=True)

    def run_script(self, *args, mode=""):
        return subprocess.run(
            [sys.executable, str(self.capture_script), "--python", sys.executable,
             "--baseline-prefix", str(self.prefix), "--output-root", str(self.output), *args],
            cwd=self.root, env={**self.env, "BANDWIDTH_TEST_MODE": mode},
            capture_output=True, text=True, timeout=30,
        )

    def captures(self):
        if not self.log.exists():
            return []
        return [args for args in map(json.loads, self.log.read_text().splitlines()) if "-o" in args]

    def test_dry_run_has_no_tool_calls_or_file_writes(self):
        result = self.run_script("--batch", "both", "--kernel", "all", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output.exists())
        self.assertEqual(result.stdout.count("# Capture:"), 10)
        self.assertEqual(result.stdout.count("# Expected grid:"), 10)
        self.assertEqual(result.stdout.count("# Export raw metrics to CSV"), 10)
        self.assertEqual(result.stdout.splitlines().count("ncu \\"), 20)
        for metric in bandwidth.METRICS.split(","):
            self.assertIn("\n    " + metric, result.stdout)

    def test_preview_bash_commands_preserve_capture_and_export_arguments(self):
        result = self.run_script("--batch", "both", "--kernel", "all", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = bandwidth.parse_args([
            "--python", sys.executable, "--baseline-prefix", str(self.prefix),
            "--output-root", str(self.output), "--batch", "both", "--kernel", "all", "--dry-run",
        ])
        session = args.output_root / "DRY_RUN_t4_attention_bandwidth"
        session.mkdir(parents=True)
        # Execute the printed Bash against fake tools, including metric joining
        # and file redirection; no actual Nsight or Python workload is launched.
        replay = subprocess.run(["bash", "-c", result.stdout], cwd=self.root,
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        expected = []
        for kernel in bandwidth.KERNELS:
            for batch in (16, 32):
                capture, export, report = bandwidth.capture_commands(args, session, batch, kernel, "ncu")
                expected.extend([capture[1:], export[1:]])
                self.assertIn("gpu__time_duration.sum", Path(f"{report}.csv").read_text())
                self.assertTrue(Path(f"{report}.export.stderr").exists())
        self.assertEqual([json.loads(line) for line in self.log.read_text().splitlines()], expected)
        for export in expected[1::2]:
            self.assertEqual(export[export.index('--page') + 1], 'raw')
            self.assertNotIn('--print-metric-name', export)

    def test_command_formatting_preserves_quoting_and_one_metrics_argument(self):
        command = [
            sys.executable, "-c", "import json, sys; print(json.dumps(sys.argv[1:]))",
            "--kernel-name", "regex:a <b> 'quoted' $(printf injected) `printf backtick`",
            "--metrics", "first,second with space,$third",
            "--baseline-run-dir", str(self.root / "path with 'quotes' $and`symbols`"),
        ]
        result = subprocess.run(["bash", "-c", bandwidth.format_shell_command(command)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), command[3:])

    def test_all_captures_reuse_baselines_and_save_validated_artifacts(self):
        result = self.run_script("--batch", "both", "--kernel", "all")
        self.assertEqual(result.returncode, 0, result.stderr)
        session, = self.output.iterdir()
        self.assertEqual((session / "status.txt").read_text(), "complete\n")
        self.assertEqual(len(list(session.glob("*.ncu-rep"))), 10)
        self.assertEqual(len(list(session.glob("*.csv"))), 10)
        self.assertEqual(len(list(session.glob("*.selection.json"))), 10)
        for selection in session.glob("*.selection.json"):
            self.assertEqual(len(json.loads(selection.read_text())["validated_workload_runs"]), 2)
        for args in self.captures():
            self.assertIn(Path(args[args.index("--baseline-run-dir") + 1]).resolve(),
                          [Path(f"{self.prefix}_b16").resolve(), Path(f"{self.prefix}_b32").resolve()])
            self.assertEqual(args[args.index("--replay-mode") + 1], "application")
            for option, expected in (
                ("--devices", "0"), ("--nvtx-include", "cached_decode_forward/"),
                ("--range-filter", "::2"), ("--kernel-name-base", "demangled"),
                ("--rename-kernels", "off"), ("--launch-count", "1"),
                ("--cache-control", "none"), ("--clock-control", "none"),
                ("--phase", "cached_decode"),
            ):
                self.assertEqual(args[args.index(option) + 1], expected)
            self.assertIn("--no-torch-profiler", args)
            self.assertIn("--output-dir", args)
            self.assertNotIn("--run-dir", args)
        for filename in ("invocation.sh", "commands.sh", "uv.lock", "git-commit.txt", "git-status.txt",
                         "gpu-before.txt", "gpu-after.txt", "ncu-version.txt", "settings.txt", "source-checks.jsonl"):
            self.assertTrue((session / filename).is_file(), filename)
        config = json.loads((session / "capture-config.json").read_text())
        self.assertEqual(config["record_origin"], "capture_startup")
        self.assertEqual(config["source_mode"], "strict")
        self.assertEqual(len(config["points"]), 10)
        for point, command in zip(config["points"], self.captures()):
            self.assertEqual(command[command.index("--kernel-name") + 1], f"regex:{point['selector_regex']}")
            self.assertEqual(int(command[command.index("--launch-skip") + 1]), point["launch_skip"])
        self.assertFalse((session / "capture-script.py").exists())
        checks = [json.loads(line) for line in (session / "source-checks.jsonl").read_text().splitlines()]
        self.assertTrue(all(check["passed"] for check in checks))
        self.assertEqual(checks[0]["stage"], "startup")
        self.assertEqual(checks[-1]["stage"], "completion")
        self.assertEqual(len(checks), 22)

    def test_dirty_checkout_fails_before_any_profiler_calls(self):
        for dirty_kind in ("unstaged", "staged", "untracked"):
            with self.subTest(dirty_kind=dirty_kind):
                self.output = self.root / f"dirty-{dirty_kind}"
                source = self.checkout / "new-source.py"
                if dirty_kind == "untracked":
                    source.write_text("# untracked source\n")
                else:
                    with self.capture_script.open("a") as handle:
                        handle.write("\n# local change\n")
                    if dirty_kind == "staged":
                        subprocess.run(["git", "add", "experiments/profile_attention_bandwidth.py"],
                                       cwd=self.checkout, check=True, capture_output=True)
                result = self.run_script("--batch", "both", "--kernel", "all")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("must be clean at startup", result.stderr)
                self.assertFalse(self.log.exists())
                session, = self.output.iterdir()
                self.assertEqual((session / "status.txt").read_text(), "failed (exit 2)\n")
                self.assertTrue((session / "git-status.txt").read_text().strip())
                subprocess.run(["git", "restore", "--source=HEAD", "--staged", "--worktree",
                                "experiments/profile_attention_bandwidth.py"],
                               cwd=self.checkout, check=True, capture_output=True)
                source.unlink(missing_ok=True)

    def test_edits_and_commits_during_collection_fail_and_stop_later_captures(self):
        for mode in ("dirty_checkout", "changed_commit"):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                self.log.unlink(missing_ok=True)
                result = self.run_script("--batch", "both", "--kernel", "all", mode=mode)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("after collection qk B=16", result.stderr)
                self.assertEqual(len(self.captures()), 1)
                session, = self.output.iterdir()
                self.assertEqual((session / "status.txt").read_text(), "failed (exit 2)\n")
                self.assertTrue((session / "qk_b16.ncu-rep").exists())
                self.assertFalse((session / "qk_b16.selection.json").exists())
                checks = [json.loads(line) for line in (session / "source-checks.jsonl").read_text().splitlines()]
                self.assertFalse(checks[-1]["passed"])
                if mode == "changed_commit":
                    self.assertNotEqual(checks[-1]["expected_commit"], checks[-1]["git_commit"])
                    self.assertFalse(checks[-1]["git_status"].strip())
                subprocess.run(["git", "restore", "experiments/profile_attention_bandwidth.py"],
                               cwd=self.checkout, check=True, capture_output=True)

    def test_changed_capture_configuration_fails_collection(self):
        result = self.run_script("--batch", "both", "--kernel", "all", mode="changed_config")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("configuration changed", result.stderr)
        self.assertEqual(len(self.captures()), 1)
        session, = self.output.iterdir()
        self.assertEqual((session / "status.txt").read_text(), "failed (exit 2)\n")
        self.assertFalse((session / "qk_b16.selection.json").exists())

    def test_saved_configuration_survives_changes_to_implementation_defaults(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        session, = self.output.iterdir()
        config = json.loads((session / "capture-config.json").read_text())
        args = bandwidth.parse_args(["--baseline-prefix", str(self.prefix)])
        with patch.object(bandwidth, "KERNELS", {}):
            command, _, report = bandwidth.capture_commands(args, session, 16, "qk", "ncu", config=config)
            self.assertEqual(command[command.index("--kernel-name") + 1], "regex:gemv2T_kernel_val")
            bandwidth.validate_capture(report, 16, "qk", config=config)

    def test_edit_during_setup_prevents_first_capture(self):
        result = self.run_script(mode="dirty_setup")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("before qk B=16", result.stderr)
        self.assertEqual(self.captures(), [])
        session, = self.output.iterdir()
        self.assertEqual((session / "status.txt").read_text(), "failed (exit 2)\n")

    def test_edit_after_last_capture_prevents_completion(self):
        gpu = self.bin / "nvidia-smi"
        gpu.write_text(f"#!{sys.executable}\n" + r'''
import os
from pathlib import Path
counter = Path(os.environ['BANDWIDTH_TEST_LOG']).with_name('gpu-call-count')
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if count == 2:
    with Path('experiments/profile_attention_bandwidth.py').open('a') as source:
        source.write('\n# edited after last capture\n')
print('fake GPU metadata')
''')
        result = self.run_script()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("at completion", result.stderr)
        self.assertNotIn("Reports saved in", result.stdout)
        session, = self.output.iterdir()
        self.assertEqual((session / "status.txt").read_text(), "failed (exit 2)\n")
        self.assertTrue((session / "qk_b16.selection.json").exists())
        checks = [json.loads(line) for line in (session / "source-checks.jsonl").read_text().splitlines()]
        self.assertEqual(checks[-1]["stage"], "completion")
        self.assertFalse(checks[-1]["passed"])

    def test_dirty_replay_metadata_rejects_capture_even_if_checkout_is_clean_again(self):
        result = self.run_script("--batch", "both", "--kernel", "all", mode="dirty_replay")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Replay did not use the clean captured commit", result.stderr)

    def test_debug_allows_dirty_source_and_propagates_to_replays(self):
        with self.capture_script.open("a") as source:
            source.write("\n# uncommitted development change\n")
        result = self.run_script("--debug", mode="dirty_checkout")
        self.assertEqual(result.returncode, 0, result.stderr)
        session, = self.output.iterdir()
        self.assertIn("_debug_", session.name)
        self.assertEqual((session / "source-mode.txt").read_text(), "debug\n")
        self.assertEqual((session / "status.txt").read_text(), "complete\n")
        self.assertTrue(all("--debug" in args for args in self.captures()))
        checks = [json.loads(line) for line in (session / "source-checks.jsonl").read_text().splitlines()]
        self.assertTrue(all(c["source_mode"] == "debug" and c["accepted"] for c in checks))
        self.assertTrue(any(not c["passed"] for c in checks))
        config = json.loads((session / "capture-config.json").read_text())
        self.assertEqual(config["source_mode"], "debug")

    def test_debug_keeps_kernel_selection_validation(self):
        result = self.run_script("--debug", mode="wrong_grid")
        self.assertNotEqual(result.returncode, 0)
        session, = self.output.iterdir()
        self.assertNotEqual((session / "status.txt").read_text(), "complete\n")
        session, = self.output.iterdir()
        self.assertEqual((session / "status.txt").read_text(), "failed (exit 2)\n")
        self.assertEqual(len(self.captures()), 1)
        self.assertFalse((session / "qk_b16.selection.json").exists())

    def test_ignored_output_files_do_not_make_checkout_dirty(self):
        self.output = self.checkout / "runs"
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_remains_available_with_local_edits(self):
        with self.capture_script.open("a") as source:
            source.write("\n# local edit\n")
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output.exists())

    def test_selectors_match_first_layer_launches_in_saved_traces(self):
        """Check real trace selection separately from fake-tool orchestration."""
        expected_sdpa_order = {"qk": 7, "kv_expand": 4, "scale": 6, "av": 14, "kv_convert": 1}
        for batch in (16, 32):
            path = SCRIPT.parents[1] / (
                f"results/published/20261010T131445Z_t4_decode_attribution_n500_profile_b{batch}/trace.json")
            events = [e for e in json.loads(path.read_text())["traceEvents"] if e.get("ph") == "X"]
            forward, = [e for e in events if e.get("cat") == "user_annotation"
                        and e["name"] == "cached_decode_forward"]
            ops = {e["args"]["External id"]: e for e in events if e.get("cat") == "cpu_op"
                   and (e["pid"], e["tid"]) == (forward["pid"], forward["tid"])
                   and forward["ts"] <= e["ts"]
                   and e["ts"] + e["dur"] <= forward["ts"] + forward["dur"] + 0.01}
            sdpa = min((e for e in ops.values() if e["name"] == "aten::scaled_dot_product_attention"),
                       key=lambda e: e["ts"])
            kernels = sorted((e for e in events if e.get("cat") == "kernel"
                              and e.get("args", {}).get("External id") in ops), key=lambda e: e["ts"])
            first_sdpa = [e for e in kernels if sdpa["ts"] <= ops[e["args"]["External id"]]["ts"]
                          and ops[e["args"]["External id"]]["ts"] + ops[e["args"]["External id"]]["dur"]
                          <= sdpa["ts"] + sdpa["dur"] + 0.01]
            for kernel, (pattern, skip, grids) in bandwidth.KERNELS.items():
                with self.subTest(batch=batch, kernel=kernel):
                    matches = [e for e in kernels if re.search(pattern, e["name"])]
                    selected = matches[skip]
                    self.assertEqual(selected, first_sdpa[expected_sdpa_order[kernel]])
                    self.assertEqual(selected["args"]["grid"], grids[batch])
                    self.assertEqual(selected["args"]["block"], [32, 4, 1] if kernel == "av" else [128, 1, 1])

    def test_long_form_csv_is_also_accepted(self):
        result = self.run_script(mode="long_csv")
        self.assertEqual(result.returncode, 0, result.stderr)
        session, = self.output.iterdir()
        self.assertTrue((session / "qk_b16.selection.json").exists())

    def test_preflight_uses_launch_stats_without_counters(self):
        result = self.run_script("--batch", "both", "--kernel", "all", "--preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.captures()), 10)
        for args in self.captures():
            self.assertEqual(args[args.index("--section") + 1], "LaunchStats")
            self.assertNotIn("--metrics", args)
            self.assertNotIn("SpeedOfLight", args)

    def test_check_session_reexports_without_launching_workloads(self):
        result = self.run_script("--kernel", "av", "--preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        session, = self.output.iterdir()
        self.log.write_text("")
        with self.capture_script.open("a") as source:
            source.write("\n# local edit during report-only validation\n")
        result = self.run_script("--kernel", "av", "--check-session", str(session))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captures(), [])
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index('--rename-kernels') + 1], 'off')

    def test_kv_expand_matches_nsight_demangled_integer_casts(self):
        pattern = bandwidth.KERNELS["kv_expand"][0]
        for name in (
            "void at::native::elementwise_kernel<128, 2, direct_copy_kernel_cuda>",
            "void at::native::elementwise_kernel<(int)128, (int)2, direct_copy_kernel_cuda>",
        ):
            self.assertIsNotNone(re.search(pattern, name))
        self.assertIsNone(re.search(pattern,
            "void at::native::elementwise_kernel<(int)128, (int)4, direct_copy_kernel_cuda>"))

    def test_reruns_create_distinct_sessions_and_details_is_selectable(self):
        for _ in range(2):
            result = self.run_script("--batch", "32", "--kernel", "qk", "--details")
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.output.iterdir())), 2)
        for args in self.captures():
            self.assertIn("MemoryWorkloadAnalysis", args)
            self.assertIn("Occupancy", args)
            self.assertNotIn("--metrics", args)

    def test_failures_retain_evidence_and_stop_later_captures(self):
        for mode in ("counter_failure", "wrong_grid", "wrong_name", "missing_report",
                     "empty_csv", "multiple_kernels", "invalid_outputs"):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                if self.log.exists():
                    self.log.unlink()
                result = self.run_script("--batch", "both", "--kernel", "all", mode=mode)
                self.assertNotEqual(result.returncode, 0)
                session, = self.output.iterdir()
                self.assertTrue((session / "status.txt").read_text().startswith("failed"))
                self.assertEqual(len(self.captures()), 1)
                self.assertTrue((session / "qk_b16.log").exists())
                self.assertFalse(list(session.glob("*.selection.json")))
                if mode == "counter_failure":
                    self.assertEqual(result.returncode, 7)
                    self.assertIn("ERR_NVGPUCTRPERM", (session / "qk_b16.log").read_text())

    def test_interpreter_symlink_is_preserved(self):
        interpreter = self.root / "venv-python"
        interpreter.symlink_to(sys.executable)
        result = self.run_script("--python", str(interpreter), "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(shlex.quote(str(interpreter)), result.stdout)

    @unittest.skipUnless(os.name == "posix", "Linux/macOS process-group interruption")
    def test_interruption_stops_workload_and_preserves_failed_session(self):
        for sig, expected_code in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=sig):
                self.output = self.root / f"signal-{sig}"
                for name in ("child-ready", "child-stopped"):
                    (self.root / name).unlink(missing_ok=True)
                process = subprocess.Popen(
                    [sys.executable, str(self.capture_script), "--python", sys.executable,
                     "--baseline-prefix", str(self.prefix), "--output-root", str(self.output),
                     "--batch", "both", "--kernel", "all"],
                    cwd=self.root, env={**self.env, "BANDWIDTH_TEST_MODE": "interrupt"},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not (self.root / "child-ready").exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("Simulated workload did not become ready")
                        time.sleep(0.02)
                    process.send_signal(sig)
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, expected_code, stdout + stderr)
                    self.assertTrue((self.root / "child-stopped").exists())
                    session, = self.output.iterdir()
                    self.assertEqual((session / "status.txt").read_text(),
                                     f"failed (exit {expected_code})\n")
                    self.assertTrue((session / "qk_b16.log").exists())
                    self.assertFalse((session / "qk_b32.log").exists())
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGTERM)
                    process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
