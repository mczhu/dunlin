"""Analyze independent fixed-context decode runs without pooling samples."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from dunlin.artifacts import summarize_saved_run

WORKLOAD_KEYS = (
    "model_id", "model_revision", "tokenizer_revision", "model_class",
    "tokenizer_class", "prompt_recipe_id", "prompt_text_sha256",
    "formatted_prompt_sha256", "input_token_ids_sha256", "input_tokens_per_request",
    "resolved_dtype", "attention_implementation", "compiled", "use_cache",
    "logits_to_keep", "decode_tokens_per_request", "decode_cache_policy",
    "timing_scope", "warmup_runs_per_phase", "measured_runs_per_phase",
    "phases", "metric_definitions_version", "metric_definitions",
)
ENVIRONMENT_KEYS = (
    "git_commit", "uv_lock_sha256", "python_version", "torch_version",
    "transformers_version", "backend", "device", "dtype", "device_name", "platform",
    "architecture", "os_release", "wsl", "system_physical_memory_bytes", "runtime_settings",
)
CUDA_KEYS = (
    "torch_cuda_build_version", "nvidia_driver_version", "dedicated_device_memory_bytes",
)
# Allow small OS-reported host RAM variations while rejecting hardware changes.
SYSTEM_MEMORY_TOLERANCE_BYTES = 1024 * 1024


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def latency_spread(samples: list[float]) -> dict[str, float]:
    """Use inclusive deciles; these describe samples, not confidence intervals."""
    if len(samples) < 2:
        raise ValueError("At least two latency samples are required for spread")
    deciles = statistics.quantiles(samples, n=10, method="inclusive")
    return {
        "median_seconds": statistics.median(samples),
        "p10_seconds": deciles[0], "p90_seconds": deciles[8],
        "min_seconds": min(samples), "max_seconds": max(samples),
    }


def analyze_decode_runs(run_dirs: list[Path], *, require_strict: bool = False) -> dict[str, Any]:
    """Validate a matched series and calculate gains between batch doublings.

    Each batch appears once. Analyze reverse-order replications as a separate
    series. Saved summaries are deliberately ignored in favor of raw samples.
    Debug inputs produce labeled development analysis unless require_strict is set.
    """
    if not run_dirs:
        raise ValueError("Select at least one completed run")
    runs = []
    seen_ids: set[str] = set()
    seen_batches: set[int] = set()
    for path in run_dirs:
        manifest = _read(path / "manifest.json")
        mode = manifest.get("source_mode", "strict")
        if mode not in ("strict", "debug"):
            raise ValueError(f"{path}: unknown source mode {mode}")
        if mode == "debug" and require_strict:
            raise ValueError(f"{path}: debug runs cannot be used for official matched scaling")
        config = _read(path / "config.resolved.json")
        environment = _read(path / "environment.json")
        if (manifest.get("experiment") != "batched_forward_phases"
                or config.get("experiment") != "batched_forward_phases"
                or not manifest.get("methodology_version")):
            raise ValueError(f"Expected a forward-phase benchmark: {path}")
        keys = ENVIRONMENT_KEYS + (CUDA_KEYS if environment.get("backend") == "cuda" else ())
        for label, record, required in (
            ("config", config, WORKLOAD_KEYS), ("environment", environment, keys),
        ):
            missing = [key for key in required if record.get(key) is None]
            if missing:
                raise ValueError(f"{path}: missing {label} provenance: {', '.join(missing)}")
        if mode == "strict" and environment.get("git_dirty") is not False:
            raise ValueError(f"{path}: clean source required for matched scaling")
        if config["decode_tokens_per_request"] != 1 or "cached_decode" not in config["phases"]:
            raise ValueError(f"{path}: expected one-token cached decode")
        summary = summarize_saved_run(path)
        run_id = manifest.get("run_id")
        batch = config["batch_size"]
        if not isinstance(run_id, str) or not run_id or run_id in seen_ids or batch in seen_batches:
            raise ValueError("Run IDs and batches must be unique; analyze replications separately")
        seen_ids.add(run_id)
        seen_batches.add(batch)
        with (path / "measurements.csv").open(newline="") as source:
            samples = [float(row["latency_seconds"]) for row in csv.DictReader(source)
                       if row["phase"] == "cached_decode"]
        point = {
            "run_id": run_id, "run_dir": str(path.resolve()),
            "started_at_utc": manifest.get("started_at_utc"),
            "batch_size": batch, "sample_count": len(samples),
            "latency_seconds": samples, **latency_spread(samples),
            "tokens_per_second": summary["decode_aggregate_tokens_per_second"],
        }
        runs.append((manifest, config, environment, point))

    baseline_manifest, baseline_config, baseline_environment, _ = runs[0]
    debug = any(run[0].get("source_mode") == "debug" for run in runs)
    keys = ENVIRONMENT_KEYS + (CUDA_KEYS if baseline_environment["backend"] == "cuda" else ())
    if debug:
        keys = tuple(key for key in keys if key not in ("git_commit", "uv_lock_sha256"))
    for manifest, config, environment, point in runs[1:]:
        differences = [f"config.{key}" for key in WORKLOAD_KEYS
                       if config[key] != baseline_config[key]]
        differences += [f"environment.{key}" for key in keys
                        if (abs(environment[key] - baseline_environment[key])
                            > SYSTEM_MEMORY_TOLERANCE_BYTES
                            if key == "system_physical_memory_bytes"
                            else environment.get(key) != baseline_environment[key])]
        if manifest["methodology_version"] != baseline_manifest["methodology_version"]:
            differences.append("methodology_version")
        if differences:
            raise ValueError(f"Incompatible run {point['run_id']}: {', '.join(differences)}")

    points = sorted((run[3] for run in runs), key=lambda point: point["batch_size"])
    intervals = []
    for lower, upper in zip(points, points[1:]):
        if upper["batch_size"] != 2 * lower["batch_size"]:
            raise ValueError("Adjacent points must double batch size; include missing points")
        gain = upper["tokens_per_second"] / lower["tokens_per_second"] - 1
        rising = upper["median_seconds"] > lower["median_seconds"]
        intervals.append({
            "lower_run_id": lower["run_id"], "upper_run_id": upper["run_id"],
            "lower_batch": lower["batch_size"], "upper_batch": upper["batch_size"],
            "throughput_gain_fraction": gain, "latency_increased": rising,
            "weakened_scaling_candidate": gain < 0.25 and rising,
            "below_plateau_threshold": gain < 0.10,
            "threshold_sensitivity": {str(threshold): gain < threshold and rising
                                      for threshold in (0.10, 0.25, 0.50)},
        })
    summary = {
        "schema_version": 1, "kind": "fixed_context_decode_scaling_analysis",
        "source_run_ids_in_supplied_order": [run[3]["run_id"] for run in runs],
        "config": baseline_config, "environment": baseline_environment,
        "points": points, "adjacent_doublings": intervals,
        "metric_definitions": {
            "tokens_per_second": "batch_size / median(latency_seconds)",
            "throughput_gain_fraction": "upper tokens_per_second / lower tokens_per_second - 1",
            "spread": "inclusive sample p10/p90; not confidence intervals",
        },
        "interpretation_limit": "Candidates require independent replication; no resource bottleneck is established.",
    }
    if debug:
        summary.update({
            "source_mode": "debug", "publication_eligible": False,
            "source_provenance": [{
                "run_id": run[3]["run_id"], "source_mode": run[0].get("source_mode", "strict"),
                "git_commit": run[2]["git_commit"], "git_dirty": run[2]["git_dirty"],
                "uv_lock_sha256": run[2]["uv_lock_sha256"],
            } for run in runs],
            "interpretation_limit": "Development analysis only; source state may differ across runs. "
                "Candidates require independent replication; no resource bottleneck is established.",
        })
    return summary
