"""Source provenance checks outside benchmark timing and profiler ranges."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug", action="store_true",
        help="Allow local source changes; label results as development-only",
    )


def check_checkout(
    stage: str, *, repo_root: Path = ROOT, expected_commit: str | None = None,
    initial_status: str = "", run_dir: Path | None = None, debug: bool = False,
) -> str:
    """Require clean tracked/untracked files and, when supplied, the same HEAD.

    Debug mode records the strict verdict without rejecting source changes.
    Ignored output/cache files are permitted. These boundary checks do not lock
    the checkout or detect every transient edit; use an isolated checkout for that.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root, check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Cannot verify experiment checkout at {stage}: {error}") from error
    expected_commit = expected_commit or commit
    passed = commit == expected_commit and not status.strip() and not initial_status.strip()
    record = json.dumps({
        "time_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "expected_commit": expected_commit, "git_commit": commit,
        "git_status": status, "passed": passed,
        "source_mode": "debug" if debug else "strict", "accepted": passed or debug,
    })
    if run_dir is not None:
        with (run_dir / "source-checks.jsonl").open("a") as handle:
            handle.write(record + "\n")
    else:
        print(f"source_check: {record}", file=sys.stderr, flush=True)
    if not debug and (initial_status.strip() or status.strip()):
        raise ValueError(f"Experiment checkout must be clean at {stage}; commit changes or use --debug")
    if not debug and commit != expected_commit:
        raise ValueError(f"Experiment commit changed at {stage}: {expected_commit} -> {commit}")
    return commit


@dataclass(frozen=True)
class SourceCheckout:
    """Record the starting revision and check source at workload boundaries."""

    commit: str
    run_dir: Path | None = None
    repo_root: Path = ROOT
    debug: bool = False

    @classmethod
    def start(
        cls, run_dir: Path | None = None, *, repo_root: Path = ROOT, debug: bool = False,
    ) -> SourceCheckout:
        commit = check_checkout("startup", run_dir=run_dir, repo_root=repo_root, debug=debug)
        return cls(commit, run_dir, repo_root, debug)

    def check(self, stage: str) -> None:
        check_checkout(stage, expected_commit=self.commit, run_dir=self.run_dir,
                       repo_root=self.repo_root, debug=self.debug)
