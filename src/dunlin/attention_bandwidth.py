"""Compare attention-kernel traffic and throughput across B=16/32 T4 decode.

Validate saved single-launch Nsight counters and replay evidence, then calculate
duration, DRAM traffic, and bandwidth ratios to assess the DRAM-bandwidth
hypothesis for the five selected kernels at N=500.
"""

from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


from dunlin.attention_capture import CAPTURE_CONFIG, KERNEL_ORDER, capture_point, read_capture_config


METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "l1tex_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str, label: str) -> float:
    try:
        number = float(value.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Expected finite nonnegative {label}: {value!r}")
    return number


def read_ncu_csv(path: Path) -> dict[str, Any]:
    """Require a wide raw-page export with units and exactly one launch."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    headers = [i for i, row in enumerate(rows)
               if {"ID", "Kernel Name", "Grid Size"} <= set(row)]
    if len(headers) != 1:
        raise ValueError(f"Expected one raw counter table: {path}")
    index = headers[0]
    header = rows[index]
    required = {*METRICS.values(), "profiler__replayer_passes"}
    if len(set(header)) != len(header) or not required <= set(header):
        raise ValueError(f"Missing metrics or duplicate columns: {path}")
    records = [dict(zip(header, row)) for row in rows[index + 1:] if any(row)]
    if any(len(row) != len(header) for row in rows[index + 1:] if any(row)):
        raise ValueError(f"Malformed counter row: {path}")
    units = [row for row in records if not row["ID"].strip()]
    data = [row for row in records if row["ID"].strip()]
    if len(units) != 1 or len(data) != 1:
        raise ValueError(f"Expected one units row and one launch: {path}")
    row, unit = data[0], units[0]
    result: dict[str, Any] = {}
    for label, metric in METRICS.items():
        number = _number(row[metric], metric)
        exported_unit = unit[metric].strip()
        if label == "duration_ns":
            factors = {"ns": 1, "us": 1_000, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
            if exported_unit not in factors or number == 0:
                raise ValueError(f"Invalid duration unit/value: {path}")
            number *= factors[exported_unit]
        elif label.endswith("bytes"):
            if exported_unit != "byte" or not number.is_integer():
                raise ValueError(f"Expected whole bytes: {path}")
            number = int(number)
        elif exported_unit != "%":
            raise ValueError(f"Expected percentage for {metric}: {path}")
        result[label] = number
    try:
        grid = ast.literal_eval(row["Grid Size"])
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"Invalid launch grid: {path}") from error
    if (not isinstance(grid, (list, tuple)) or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)):
        raise ValueError(f"Invalid launch grid: {path}")
    replay_passes = _number(row["profiler__replayer_passes"], "replay passes")
    if not replay_passes.is_integer() or replay_passes == 0:
        raise ValueError(f"Expected a positive whole replay count: {path}")
    result.update({
        "kernel_name": row["Kernel Name"], "kernel_id": row["ID"], "grid": list(grid),
        "block": row.get("Block Size"),
        "replay_passes": int(replay_passes),
        "dram_total_bytes": result["dram_read_bytes"] + result["dram_write_bytes"],
        "duration_us": result["duration_ns"] / 1_000,
    })
    result["dram_gb_per_second"] = result["dram_total_bytes"] / result["duration_ns"]
    return result


def compare_batches(lower: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    if (lower["kernel"] != upper["kernel"]
            or lower["batch_size"] != 16 or upper["batch_size"] != 32):
        raise ValueError("Expected a same-kernel B=16/32 pair")
    if lower["duration_ns"] <= 0 or lower["dram_total_bytes"] <= 0:
        raise ValueError("Ratios require positive B=16 duration and DRAM traffic")
    return {
        "kernel": lower["kernel"],
        "duration_ratio": upper["duration_ns"] / lower["duration_ns"],
        "dram_bytes_ratio": upper["dram_total_bytes"] / lower["dram_total_bytes"],
        "dram_bandwidth_ratio": upper["dram_gb_per_second"] / lower["dram_gb_per_second"],
    }


def analyze_session(session: Path, *, require_strict: bool = False) -> dict[str, Any]:
    """Validate every retained replay; require strict source for official evidence."""
    mode_file = session / "source-mode.txt"
    mode = mode_file.read_text().strip() if mode_file.is_file() else "strict"
    if mode not in ("strict", "debug"):
        raise ValueError(f"Unknown session source mode: {mode}")
    debug = mode == "debug"
    if debug and require_strict:
        raise ValueError("Debug sessions cannot be used for official bandwidth study analysis")
    if (session / "status.txt").read_text().strip() != "complete":
        raise ValueError("Session did not complete")
    settings = dict(line.split("=", 1) for line in (session / "settings.txt").read_text().splitlines())
    if any(settings.get(key) != value for key, value in
           {"batch": "both", "kernel": "all", "preflight": "False"}.items()):
        raise ValueError("Expected a both/all counter session, without preflight")
    inputs: dict[str, str] = {}

    def retain(path: Path) -> None:
        inputs[str(path.relative_to(session))] = sha256(path)

    def read(path: Path) -> dict[str, Any]:
        retain(path)
        return json.loads(path.read_text())

    metadata_files = ("status.txt", "settings.txt", "invocation.sh", "commands.sh",
                      "uv.lock", "git-commit.txt", "git-status.txt", "started-at.txt", "finished-at.txt",
                      "ncu-version.txt", "gpu-before.txt", "gpu-after.txt")
    metadata = {}
    for name in metadata_files:
        retain(session / name)
        if name != "uv.lock":
            metadata[name] = (session / name).read_text()
    for name in ("source-mode.txt", "source-checks.jsonl"):
        if (session / name).is_file():
            retain(session / name)
    capture_commit = metadata["git-commit.txt"].strip()
    if not debug and metadata["git-status.txt"].strip():
        raise ValueError("Capture checkout was dirty")
    if not debug and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", capture_commit):
        raise ValueError("Expected a full capture commit hash in git-commit.txt")
    capture_config = read_capture_config(session)
    retain(session / CAPTURE_CONFIG)
    if capture_config["source_mode"] != mode:
        raise ValueError("Capture configuration source mode disagrees with session")
    if capture_config["preflight"]:
        raise ValueError("Expected a counter session, without preflight")
    expected_points = [(kernel, batch) for kernel in KERNEL_ORDER for batch in (16, 32)]
    if [(p["kernel"], p["batch_size"]) for p in capture_config["points"]] != expected_points:
        raise ValueError("Expected ordered both/all capture points")
    migration = capture_config.get("migration")
    if migration is not None:
        if migration["source_git_commit"] != capture_commit:
            raise ValueError("Migration source commit disagrees with capture")
        for name, digest in migration["input_sha256"].items():
            # Migration inputs are explicit artifacts, never external paths.
            if name not in metadata_files or sha256(session / name) != digest:
                raise ValueError("Migration input provenance mismatch")
    capture_provenance = {
        "git_commit": capture_commit, "source_mode": mode,
        "configuration_origin": capture_config["record_origin"],
        "selector_validation": "recorded capture configuration",
    }
    expected_environment = None
    recorded_environment = None
    expected_config = None
    configs_by_batch: dict[int, dict[str, Any]] = {}
    captures = []
    excluded_config_keys = {"batch_size", "baseline_run_dir", "source_config", "output_root"}
    provenance_keys = ("model_id", "model_revision", "tokenizer_revision", "input_token_ids_sha256",
                       "resolved_dtype", "resolved_device", "attention_implementation",
                       "decode_cache_policy", "warmup_forwards", "validation_rtol", "validation_atol")
    for kernel in KERNEL_ORDER:
        for batch in (16, 32):
            stem = f"{kernel}_b{batch}"
            path = session / f"{stem}.csv"
            retain(path)
            capture = read_ncu_csv(path)
            selection = read(session / f"{stem}.selection.json")
            if (selection.get("selection_validated") is not True
                    or selection.get("batch_size") != batch
                    or selection.get("kernel_name") != capture["kernel_name"]
                    or selection.get("kernel_id") != capture["kernel_id"]
                    or selection.get("grid") != capture["grid"]):
                raise ValueError(f"Kernel selection mismatch: {stem}")
            point = capture_point(capture_config, kernel, batch)
            pattern, skip = point["selector_regex"], point["launch_skip"]
            if capture["grid"] != point["expected_grid"] or not re.search(pattern, capture["kernel_name"]):
                raise ValueError(f"Kernel selection mismatch: {stem}")
            workloads = sorted((session / f"{stem}_workloads").iterdir())
            recorded_workloads = [Path(name).name for name in selection["validated_workload_runs"]]
            if (not workloads or [w.name for w in workloads] != recorded_workloads
                    or len(workloads) != capture["replay_passes"]):
                raise ValueError(f"Replay records disagree: {stem}")
            replay_checks = []
            for workload in workloads:
                manifest = read(workload / "manifest.json")
                validation = read(workload / "validation.json")
                config = read(workload / "config.resolved.json")
                environment = read(workload / "environment.json")
                if (manifest.get("status") != "complete"
                        or manifest.get("source_mode", "strict") != mode
                        or manifest.get("experiment") != "profile_forward_phases"
                        or manifest.get("failure") is not None
                        or manifest.get("run_id") != workload.name
                        or manifest.get("linked_baseline_run_id") != Path(config["baseline_run_dir"]).name
                        or validation.get("forward_matches_reference") is not True
                        or validation.get("logits_finite") is not True
                        or validation.get("logits_shape") != [batch, 1, 151936]
                        or validation.get("cache_length_before") != 500
                        or validation.get("cache_length_after") != 501):
                    raise ValueError(f"Replay output validation failed: {workload}")
                baseline = capture_config["workload"]["baseline_runs"][str(batch)]
                if Path(config["baseline_run_dir"]).name != Path(baseline).name:
                    raise ValueError(f"Replay baseline disagrees with capture configuration: {workload}")
                if (config.get("batch_size") != batch or config.get("phase") != "cached_decode"
                        or config.get("input_tokens_per_request") != 500
                        or config.get("torch_profiler_enabled") is not False
                        or any(config.get(key) is None for key in provenance_keys)):
                    raise ValueError(f"Unexpected replay configuration/source: {workload}")
                if not debug and (environment.get("git_commit") != capture_commit
                                  or environment.get("git_dirty") is not False
                                  or environment.get("uv_lock_sha256") != inputs["uv.lock"]):
                    raise ValueError(f"Unexpected replay configuration/source: {workload}")
                common_config = {k: v for k, v in config.items() if k not in excluded_config_keys}
                common_environment = {
                    k: v for k, v in environment.items()
                    if not debug or k not in ("git_commit", "git_dirty", "uv_lock_sha256")
                }
                if expected_config is None:
                    expected_config, expected_environment = common_config, common_environment
                    recorded_environment = environment
                if common_config != expected_config or common_environment != expected_environment:
                    raise ValueError(f"Incompatible replay workload/environment: {workload}")
                if batch in configs_by_batch:
                    first = configs_by_batch[batch]
                    if any(config[key] != first[key] for key in ("baseline_run_dir", "source_config")):
                        raise ValueError(f"Incompatible linked baseline: {workload}")
                configs_by_batch.setdefault(batch, config)
                replay_check = {"manifest": manifest, "validation": validation}
                if debug:
                    replay_check["environment"] = environment
                replay_checks.append(replay_check)
            log = session / f"{stem}.log"
            retain(log)
            warnings = [line for line in log.read_text().splitlines() if "==WARNING==" in line]
            capture.update({"kernel": kernel, "batch_size": batch, "capture_order": len(captures) + 1,
                            "sample_count": 1, "launch_skip": skip, "selector_regex": pattern,
                            "replay_checks": replay_checks, "warnings": warnings,
                            "report_retained": (session / f"{stem}.ncu-rep").is_file()})
            captures.append(capture)
    start = datetime.fromisoformat(metadata["started-at.txt"].strip())
    finish = datetime.fromisoformat(metadata["finished-at.txt"].strip())
    summary = {
        "schema_version": 2, "kind": "attention_bandwidth_analysis",
        "source_session_id": session.name, "source_session_path": str(session.resolve()),
        "capture_metadata": metadata, "environment": recorded_environment,
        "capture_provenance": capture_provenance, "capture_config": capture_config,
        "configs_by_batch": configs_by_batch, "source_sha256": inputs,
        "session_elapsed_seconds": (finish - start).total_seconds(),
        "captures": captures,
        "comparisons": [compare_batches(captures[i], captures[i + 1]) for i in range(0, 10, 2)],
        "metric_definitions": {
            "dram_gb_per_second": "(dram read bytes + dram write bytes) / duration_ns; decimal GB/s",
            "ratios": "B=32 / B=16, within the same selected kernel",
            "sample_count": "One reported launch per kernel/batch; replay passes are not independent samples",
            "throughput_pct": "Nsight sustained-peak throughput rollups; not whole-GPU busy time",
            "l2_hit_pct": "L2 sector hit rate; distinct from L2 throughput utilization",
            "scope": "Five selected launches in the second cached_decode_forward NVTX range, per batch",
        },
    }
    if debug:
        summary.update({
            "source_mode": "debug", "publication_eligible": False,
            "interpretation_limit": "Development analysis only. Recorded commits do not identify local edits; "
                "selectors are checked against recorded configuration, and replay source state may differ.",
        })
    return summary
