#!/usr/bin/env python3
"""Capture selected T4 attention kernels with Nsight Compute.

Orchestration uses only the standard library. Inference and output validation
stay in profile_forward_phases.py; bandwidth analysis is a separate step.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
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


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
# Keep this stdlib-only orchestrator usable without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))
from dunlin.source import add_source_arguments, check_checkout
from dunlin.attention_capture import (
    COUNTER_METRICS, capture_point, read_capture_config, validate_capture_config, write_capture_config,
)

METRICS = ",".join(COUNTER_METRICS)
# Selectors and grids come from the validated B=16/32, N=500 T4 traces.
# Kernel name/grid checks cannot establish layer identity if launch order changes.
KERNELS = {
    "qk": ("gemv2T_kernel_val", 0, {16: [63, 1, 224], 32: [63, 1, 448]}),
    "kv_expand": (
        r"elementwise_kernel<(?:\(int\))?128, (?:\(int\))?2,.*direct_copy_kernel_cuda", 0,
        {16: [28056, 1, 1], 32: [56112, 1, 1]},
    ),
    "scale": (
        "AUnaryFunctor<float.*MulFunctor<float", 3,
        {16: [7014, 1, 1], 32: [14028, 1, 1]},
    ),
    "av": ("internal::gemvx::kernel", 0, {16: [2, 1, 224], 32: [2, 1, 448]}),
    # First large SDPA input conversion: four other forward copies and the
    # small query conversion precede it. See decode_execution_kernel_ranking.
    "kv_convert": (
        "unrolled_elementwise_kernel<.*direct_copy_kernel_cuda", 5,
        {16: [2004, 1, 1], 32: [4008, 1, 1]},
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("16", "32", "both"), default="16")
    parser.add_argument("--kernel", choices=(*KERNELS, "all"), default="qk",
                        help="One configured selection; all collects all five, not every forward kernel")
    parser.add_argument("--details", action="store_true",
                        help="Collect throughput, memory, and occupancy sections")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate selections using LaunchStats only, without bandwidth counters")
    parser.add_argument("--check-session", type=Path,
                        help="Re-export and validate existing reports only; never launch a workload")
    parser.add_argument("--baseline-prefix", type=Path,
                        default=REPO_ROOT / "results/published/20261009T215054Z_t4_decode_n500",
                        help="Existing timing-run prefix, before _b16/_b32")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs",
                        help="Parent of a new unique session directory")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Workload interpreter; default: the current Python executable")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without tools, CUDA, or file writes")
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    # Resolve caller-supplied paths before running child commands at the repo root.
    for name in ("baseline_prefix", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Keep a venv interpreter's symlink path: resolving it can select system Python.
    args.python = args.python.expanduser().absolute()
    return args


def resolve_capture_config(args: argparse.Namespace, batches: list[int], kernels: list[str]) -> dict:
    """Resolve once; commands and validation consume this same recorded data."""
    sections = (["LaunchStats", "SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy"]
                if args.details and not args.preflight else ["LaunchStats"])
    config = {
        "schema_version": 1, "kind": "attention_bandwidth_capture",
        "source_mode": "debug" if args.debug else "strict", "record_origin": "capture_startup",
        "preflight": args.preflight,
        "capture": {
            "device": "0", "nvtx_include": "cached_decode_forward/", "range_filter": "::2",
            "kernel_name_base": "demangled", "rename_kernels": "off", "launch_count": 1,
            "replay_mode": "application", "cache_control": "none", "clock_control": "none",
            "sections": sections, "metrics": [] if args.preflight or args.details else list(COUNTER_METRICS),
        },
        "workload": {
            "python": str(args.python), "runner": "experiments/profile_forward_phases.py",
            "phase": "cached_decode", "torch_profiler_enabled": False,
            "baseline_runs": {str(batch): f"{args.baseline_prefix}_b{batch}" for batch in batches},
        },
        "points": [{"kernel": kernel, "batch_size": batch, "selector_regex": KERNELS[kernel][0],
                    "launch_skip": KERNELS[kernel][1], "expected_grid": KERNELS[kernel][2][batch]}
                   for kernel in kernels for batch in batches],
    }
    return validate_capture_config(config)


def capture_commands(
    args: argparse.Namespace, session: Path, batch: int, kernel: str, ncu: str,
    *, config: dict | None = None,
) -> tuple[list[str], list[str], Path]:
    # The optional resolution is for dry-run callers. Real captures pass their
    # startup configuration; later implementation defaults cannot replace it.
    if config is None:
        batches = [16, 32] if args.batch == "both" else [int(args.batch)]
        kernels = list(KERNELS) if args.kernel == "all" else [args.kernel]
        config = resolve_capture_config(args, batches, kernels)
    point = capture_point(config, kernel, batch)
    settings, workload = config["capture"], config["workload"]
    report = session / f"{kernel}_b{batch}"
    collection = [part for section in settings["sections"] for part in ("--section", section)]
    if settings["metrics"]:
        collection += ["--metrics", ",".join(settings["metrics"])]
    command = [
        ncu, "--devices", settings["device"], "--nvtx", "--nvtx-include", settings["nvtx_include"],
        "--range-filter", settings["range_filter"], "--kernel-name-base", settings["kernel_name_base"],
        "--rename-kernels", settings["rename_kernels"], "--kernel-name", f"regex:{point['selector_regex']}",
        "--launch-skip", str(point["launch_skip"]), "--launch-count", str(settings["launch_count"]),
        "--replay-mode", settings["replay_mode"], "--cache-control", settings["cache_control"],
        "--clock-control", settings["clock_control"],
        *collection, "-o", str(report), workload["python"], workload["runner"],
        "--baseline-run-dir", workload["baseline_runs"][str(batch)],
        "--phase", workload["phase"], "--no-torch-profiler", "--output-dir", f"{report}_workloads",
        *(["--debug"] if config["source_mode"] == "debug" else []),
    ]
    return command, export_command(ncu, report), report


def export_command(ncu: str, report: Path) -> list[str]:
    return [ncu, "--import", f"{report}.ncu-rep", "--page", "raw", "--csv",
            "--print-units", "base", "--rename-kernels", "off"]


def check_session(session: Path, batches: list[int], kernels: list[str]) -> None:
    config = read_capture_config(session)
    ncu = shutil.which("ncu")
    if not ncu:
        raise ValueError("ncu is missing; install the Nsight Compute CLI")
    failures = []
    for kernel in kernels:
        for batch in batches:
            report = session / f"{kernel}_b{batch}"
            if not Path(f"{report}.ncu-rep").is_file():
                failures.append(f"{kernel} B={batch}: no retained report")
                continue
            try:
                run_command(export_command(ncu, report), Path(f"{report}.csv"),
                            stderr=Path(f"{report}.export.stderr"))
                validate_capture(report, batch, kernel, config=config)
                print(f"{kernel} B={batch}: passed", flush=True)
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                failures.append(f"{kernel} B={batch}: {error}")
    if failures:
        raise ValueError("Session checks failed:\n" + "\n".join(failures))


def format_shell_command(command: list[str], *, application_index: int | None = None) -> str:
    """Render copyable Bash with one option per line and exact argument quoting."""
    groups: list[list[str]] = [[]]
    for index, argument in enumerate(command):
        if groups[-1] and (argument.startswith("--") or argument == "-o"
                           or index == application_index):
            groups.append([])
        groups[-1].append(argument)
    lines = []
    for index, group in enumerate(groups):
        if group[0] == "--metrics" and len(group) == 2:
            # printf joins the pieces into ONE argument, without inserted whitespace.
            metrics = group[1].split(",")
            pieces = [metric + "," for metric in metrics[:-1]] + [metrics[-1]]
            parts = ["    " + shlex.quote(piece) for piece in pieces]
            lines.append("  --metrics \"$(printf '%s' \\\n" + " \\\n".join(parts) + ")\"")
        else:
            lines.append(("" if index == 0 else "  ") + shlex.join(group))
    return " \\\n".join(lines)


def stop_process(process: subprocess.Popen) -> None:
    """Interrupt the whole Nsight/workload group, then bound the shutdown wait."""
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGINT:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        # Kill any remaining descendants even if Nsight exited before its workload.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return


def run_command(
    command: list[str], output: Path, *, stderr: Path | None = None,
    stream: bool = False, timeout: float | None = None,
) -> None:
    """Run argument lists without a shell; optionally stream a retained capture log."""
    with ExitStack() as files:
        log = files.enter_context(output.open("w"))
        errors = files.enter_context(stderr.open("w")) if stderr else subprocess.STDOUT
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE if stream else log,
            stderr=errors, text=True, encoding="utf-8", errors="replace",
            start_new_session=os.name == "posix",
        )
        try:
            if stream:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            returncode = process.wait(timeout=timeout)
        except BaseException:
            stop_process(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def validate_capture(
    report: Path, batch: int, kernel: str, *, config: dict, expected_commit: str | None = None,
) -> None:
    point = capture_point(config, kernel, batch)
    pattern, expected_grid = point["selector_regex"], point["expected_grid"]
    workloads = sorted(Path(f"{report}_workloads").iterdir())
    if not workloads:
        raise ValueError("No workload artifacts from application replay")
    for workload in workloads:
        manifest = json.loads((workload / "manifest.json").read_text())
        validation = json.loads((workload / "validation.json").read_text())
        if expected_commit is not None:
            environment = json.loads((workload / "environment.json").read_text())
            if (environment.get("git_commit") != expected_commit
                    or manifest.get("source_mode", "strict") != "strict"
                    or environment.get("git_dirty") is not False):
                raise ValueError(f"Replay did not use the clean captured commit: {workload}")
        if (manifest.get("status") != "complete"
                or validation.get("forward_matches_reference") is not True):
            raise ValueError(f"Counter workload did not complete output validation: {workload}")
        if (validation.get("logits_shape") != [batch, 1, 151936]
                or validation.get("logits_finite") is not True
                or validation.get("cache_length_before") != 500
                or validation.get("cache_length_after") != 501):
            raise ValueError(f"Counter workload has unexpected output shape or cache lengths: {workload}")
    with Path(f"{report}.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next((i for i, row in enumerate(rows)
                         if {"ID", "Kernel Name", "Grid Size"} <= set(row)), None)
    if header_index is None:
        raise ValueError("No kernel table in exported CSV; inspect the Nsight report/log")
    header = rows[header_index]
    # Raw CSV has one column per metric and an initial units row with no ID.
    # Also accept long-form exports with one row per metric.
    records = [dict(zip(header, row)) for row in rows[header_index + 1:]
               if len(row) == len(header) and row != header
               and row[header.index("ID")].strip()]
    if not records or len({r["ID"] for r in records}) != 1:
        raise ValueError("Expected exactly one profiled kernel in exported CSV")
    for record in records:
        if not re.search(pattern, record["Kernel Name"]):
            raise ValueError(f"Unexpected kernel name {record['Kernel Name']!r}; expected regex {pattern!r}")
        grid = [int(value) for value in re.findall(r"\d+", record["Grid Size"])]
        if grid != expected_grid:
            raise ValueError("Unexpected kernel grid; inspect selection before proceeding")
    Path(f"{report}.selection.json").write_text(json.dumps({
        "batch_size": batch, "kernel_name": records[0]["Kernel Name"],
        "kernel_id": records[0]["ID"], "grid": expected_grid,
        "selection_validated": True, "validated_workload_runs": [str(p) for p in workloads],
    }, indent=2) + "\n")
    print("Output validation and single-kernel selection passed", flush=True)


def check_capture_checkout(session: Path, stage: str) -> None:
    """Check source at capture boundaries; retain each check with the session."""
    check_checkout(
        stage, repo_root=REPO_ROOT, run_dir=session,
        expected_commit=(session / "git-commit.txt").read_text().strip(),
        initial_status=(session / "git-status.txt").read_text(),
        debug=(session / "source-mode.txt").read_text().strip() == "debug",
    )


def prepare_session(args: argparse.Namespace, session: Path, batches: list[int], config: dict) -> str:
    (session / "source-mode.txt").write_text("debug\n" if args.debug else "strict\n")
    (session / "invocation.sh").write_text(shlex.join([sys.executable, str(SCRIPT), *sys.argv[1:]]) + "\n")
    shutil.copyfile(REPO_ROOT / "uv.lock", session / "uv.lock")
    run_command(["git", "rev-parse", "HEAD"], session / "git-commit.txt")
    run_command(["git", "status", "--porcelain", "--untracked-files=all"], session / "git-status.txt")
    check_capture_checkout(session, "startup")
    write_capture_config(session, config)
    (session / "settings.txt").write_text(
        f"baseline_prefix={args.baseline_prefix}\npython={args.python}\nbatch={args.batch}\n"
        f"kernel={args.kernel}\ndetails={args.details}\npreflight={args.preflight}\nmetrics={METRICS}\n"
    )
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ValueError(f"Python is missing or not executable: {args.python}; run uv sync --locked")
    for batch in batches:
        for filename in ("manifest.json", "config.resolved.json"):
            path = Path(f"{args.baseline_prefix}_b{batch}") / filename
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing baseline: {path}")
    ncu = shutil.which("ncu")
    if not ncu:
        raise ValueError("ncu is missing; install the Nsight Compute CLI")
    gpu_tool = shutil.which("nvidia-smi")
    if not gpu_tool:
        raise ValueError("nvidia-smi is missing")
    (session / "nvidia-smi-path.txt").write_text(gpu_tool + "\n")
    run_command([ncu, "--version"], session / "ncu-version.txt")
    run_command([gpu_tool, "-q"], session / "gpu-before.txt", stderr=session / "gpu-before.stderr")
    run_command([ncu, "--help"], session / "ncu-help.txt")
    run_command([ncu, "--list-sections"], session / "ncu-sections.txt")
    run_command([
        ncu, "--query-metrics", "--devices", "0", "--query-metrics-mode", "suffix",
        "--metrics", "dram__bytes_read,dram__bytes_write,dram__throughput,lts__throughput,"
        "lts__t_sector_hit_rate,l1tex__throughput,sm__throughput",
    ], session / "available-metrics.txt")
    return ncu


def check_capture_config(session: Path, expected: dict) -> None:
    if read_capture_config(session) != expected:
        raise ValueError("Recorded capture configuration changed during collection")


def capture(args: argparse.Namespace, session: Path, batch: int, kernel: str, ncu: str, config: dict) -> None:
    command, export, report = capture_commands(args, session, batch, kernel, ncu, config=config)
    if args.dry_run:
        print(f"\n# Capture: {kernel}, B={batch}")
        print(f"# Expected grid: {capture_point(config, kernel, batch)['expected_grid']}")
        application_index = command.index("experiments/profile_forward_phases.py") - 1
        print(format_shell_command(command, application_index=application_index))
        print("\n# Export raw metrics to CSV")
        print(format_shell_command(export) + " \\\n"
              + f"  > {shlex.quote(str(report) + '.csv')} \\\n"
              + f"  2> {shlex.quote(str(report) + '.export.stderr')}")
        return
    check_capture_config(session, config)
    check_capture_checkout(session, f"before {kernel} B={batch}")
    print(f"Profiling {kernel} at B={batch}", flush=True)
    with (session / "commands.sh").open("a") as commands:
        commands.write(shlex.join(command) + "\n")
    print(shlex.join(command), flush=True)
    run_command(command, Path(f"{report}.log"), stream=True)
    check_capture_config(session, config)
    check_capture_checkout(session, f"after collection {kernel} B={batch}")
    report_file = Path(f"{report}.ncu-rep")
    if not report_file.is_file() or not report_file.stat().st_size:
        raise ValueError(f"No report produced for {kernel} B={batch}")
    with (session / "commands.sh").open("a") as commands:
        commands.write(shlex.join(export) + "\n")
    run_command(export, Path(f"{report}.csv"), stderr=Path(f"{report}.export.stderr"))
    validate_capture(report, batch, kernel, config=config,
                     expected_commit=None if args.debug else (session / "git-commit.txt").read_text().strip())


def write_time(path: Path) -> None:
    path.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ\n"))


def main() -> int:
    args = parse_args()
    batches = [16, 32] if args.batch == "both" else [int(args.batch)]
    kernels = list(KERNELS) if args.kernel == "all" else [args.kernel]
    if args.check_session:
        try:
            check_session(args.check_session.expanduser().resolve(), batches, kernels)
        except (OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        return 0
    config = resolve_capture_config(args, batches, kernels)
    if args.dry_run:
        session = args.output_root / "DRY_RUN_t4_attention_bandwidth"
        print("# Dry run: no profiler commands executed or artifacts created.")
        print("# Output paths below use a placeholder; normal runs create a unique session.")
        print(f"cd {shlex.quote(str(REPO_ROOT))}")
        for kernel in kernels:
            for batch in batches:
                capture(args, session, batch, kernel, "ncu", config)
        return 0

    # SIGTERM follows the same artifact-preserving shutdown path as Ctrl+C.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "t4_attention_bandwidth_debug" if args.debug else "t4_attention_bandwidth"
    session = Path(tempfile.mkdtemp(prefix=f"{timestamp}_{label}_", dir=args.output_root))
    print(f"Session: {session}", flush=True)
    (session / "status.txt").write_text("running\n")
    write_time(session / "started-at.txt")
    exit_code = 1
    try:
        ncu = prepare_session(args, session, batches, config)
        for kernel in kernels:
            for batch in batches:
                capture(args, session, batch, kernel, ncu, config)
        exit_code = 0
    except KeyboardInterrupt:
        exit_code = 130
        print("Capture interrupted", file=sys.stderr)
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except subprocess.CalledProcessError as error:
        exit_code = error.returncode if error.returncode > 0 else 128 - error.returncode
        print(f"Command failed (exit {exit_code}): {shlex.join(error.cmd)}", file=sys.stderr)
    except (OSError, ValueError) as error:
        exit_code = 2
        print(f"Error: {error}", file=sys.stderr)
    finally:
        try:
            run_command(["nvidia-smi", "-q"], session / "gpu-after.txt",
                        stderr=session / "gpu-after.stderr", timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            with (session / "gpu-after.stderr").open("a") as log:
                log.write(f"{error}\n")
        if exit_code == 0:
            try:
                check_capture_config(session, config)
                check_capture_checkout(session, "completion")
            except (OSError, ValueError) as error:
                exit_code = 2
                print(f"Error: {error}", file=sys.stderr)
        write_time(session / "finished-at.txt")
        status = "complete" if exit_code == 0 else f"failed (exit {exit_code})"
        (session / "status.txt").write_text(status + "\n")
        if exit_code:
            print(f"Capture failed; artifacts retained in {session}", file=sys.stderr)
    if exit_code == 0:
        print(f"Reports saved in {session}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
