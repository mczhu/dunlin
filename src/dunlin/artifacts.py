"""Create and read self-describing benchmark run artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
FORWARD_METRIC_DEFINITIONS_VERSION = "batched_forward_phase_metrics_v1"
FORWARD_METRIC_DEFINITIONS = {
    "median_prefill_forward_latency_seconds": {
        "formula": "median(prefill latency_seconds)",
        "unit": "seconds",
    },
    "median_cached_decode_forward_latency_seconds": {
        "formula": "median(cached_decode latency_seconds)",
        "unit": "seconds",
    },
    "prefill_aggregate_prompt_tokens_per_second": {
        "formula": "batch_size * input_tokens_per_request / median_prefill_forward_latency_seconds",
        "unit": "prompt tokens/second",
    },
    "decode_aggregate_tokens_per_second": {
        "formula": "batch_size / median_cached_decode_forward_latency_seconds",
        "unit": "decoded tokens/second",
    },
}


class ForwardPhaseSummary(TypedDict, total=False):
    schema_version: int
    metric_definitions_version: str
    prefill_sample_count: int
    decode_sample_count: int
    median_prefill_forward_latency_seconds: float
    median_cached_decode_forward_latency_seconds: float
    prefill_aggregate_prompt_tokens_per_second: float
    decode_aggregate_tokens_per_second: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_run_dir(
    experiment: str, requested: Path | None = None, output_root: Path | None = None,
    *, debug: bool = False,
) -> Path:
    if requested is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = f"{experiment}_debug" if debug else experiment
        requested = (output_root or ROOT / "runs") / f"{timestamp}_{label}_{uuid.uuid4().hex[:8]}"
    requested = requested.resolve()
    requested.mkdir(parents=True, exist_ok=False)
    return requested


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def new_manifest(experiment: str, run_id: str, *, debug: bool = False) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": "benchmark",
        "source_mode": "debug" if debug else "strict",
        "experiment": experiment,
        "methodology_version": f"{experiment}_v1",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "failure": None,
        "linked_baseline_run_id": None,
        "artifacts": [],
    }


def phase_summary(
    prefill_latencies: list[float],
    decode_latencies: list[float],
    batch_size: int,
    input_tokens: int,
    phases: tuple[str, ...] = ("prefill", "cached_decode"),
) -> ForwardPhaseSummary:
    """Summarize measured forward latencies and compute aggregate throughput."""
    if not phases or len(set(phases)) != len(phases) or any(
        phase not in ("prefill", "cached_decode") for phase in phases
    ):
        raise ValueError("Invalid selected phases")
    samples = {"prefill": prefill_latencies, "cached_decode": decode_latencies}
    if any(bool(values) != (phase in phases) for phase, values in samples.items()):
        raise ValueError("Selected phases require samples; unselected phases must have none")
    if batch_size <= 0 or input_tokens <= 0:
        raise ValueError("Batch size and input tokens must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in prefill_latencies + decode_latencies):
        raise ValueError("Measured latencies must be finite and positive")
    summary: ForwardPhaseSummary = {
        "schema_version": SCHEMA_VERSION,
        "metric_definitions_version": FORWARD_METRIC_DEFINITIONS_VERSION,
        "prefill_sample_count": len(prefill_latencies),
        "decode_sample_count": len(decode_latencies),
    }
    if prefill_latencies:
        prefill_median = statistics.median(prefill_latencies)
        summary["median_prefill_forward_latency_seconds"] = prefill_median
        summary["prefill_aggregate_prompt_tokens_per_second"] = batch_size * input_tokens / prefill_median
    if decode_latencies:
        decode_median = statistics.median(decode_latencies)
        summary["median_cached_decode_forward_latency_seconds"] = decode_median
        summary["decode_aggregate_tokens_per_second"] = batch_size / decode_median
    return summary


def write_measurements(
    path: Path,
    run_id: str,
    batch_size: int,
    input_tokens: int,
    prefill_latencies: list[float],
    decode_latencies: list[float],
    prefill_cache_lengths: list[int],
    decode_cache_lengths: list[int],
) -> None:
    if batch_size <= 0 or input_tokens <= 0:
        raise ValueError("Batch size and input tokens must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in prefill_latencies + decode_latencies):
        raise ValueError("Measured latencies must be finite and positive")
    if len(decode_latencies) != len(prefill_cache_lengths) or len(decode_latencies) != len(decode_cache_lengths):
        raise ValueError("Decode samples and cache-length records must align")
    fields = (
        "run_id", "phase", "repetition_index", "batch_size",
        "input_tokens_per_request", "output_tokens_per_request",
        "prefill_cache_length", "decode_cache_length", "latency_seconds",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, latency in enumerate(prefill_latencies):
            writer.writerow({
                "run_id": run_id, "phase": "prefill", "repetition_index": index,
                "batch_size": batch_size, "input_tokens_per_request": input_tokens,
                "output_tokens_per_request": 0, "latency_seconds": latency,
            })
        for index, latency in enumerate(decode_latencies):
            writer.writerow({
                "run_id": run_id, "phase": "cached_decode", "repetition_index": index,
                "batch_size": batch_size, "input_tokens_per_request": input_tokens,
                "output_tokens_per_request": 1,
                "prefill_cache_length": prefill_cache_lengths[index],
                "decode_cache_length": decode_cache_lengths[index],
                "latency_seconds": latency,
            })


def summarize_saved_run(run_dir: Path) -> ForwardPhaseSummary:
    """Rebuild a completed benchmark summary using only its recorded samples."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["run_kind"] != "benchmark":
        raise ValueError("Only completed benchmark runs can be summarized")
    config = json.loads((run_dir / "config.resolved.json").read_text())
    if config.get("metric_definitions_version") not in (None, FORWARD_METRIC_DEFINITIONS_VERSION):
        raise ValueError("Unsupported metric definitions version")
    if "metric_definitions" in config and config["metric_definitions"] != FORWARD_METRIC_DEFINITIONS:
        raise ValueError("Run metric definitions differ from the supported definitions")
    with (run_dir / "measurements.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if any(row["run_id"] != manifest["run_id"] for row in rows):
        raise ValueError("Measurement run ID does not match the manifest")
    if any(
        int(row["batch_size"]) != config["batch_size"]
        or int(row["input_tokens_per_request"]) != config["input_tokens_per_request"]
        for row in rows
    ):
        raise ValueError("Measurement workload does not match the resolved config")
    phases = {
        phase: [float(row["latency_seconds"]) for row in rows if row["phase"] == phase]
        for phase in ("prefill", "cached_decode")
    }
    if len(rows) != sum(len(samples) for samples in phases.values()):
        raise ValueError("Unknown phase in measurement data")
    expected = config["measured_runs_per_phase"]
    selected_phases = tuple(config.get("phases", ("prefill", "cached_decode")))
    if any(len(samples) != (expected if phase in selected_phases else 0) for phase, samples in phases.items()):
        raise ValueError("Incomplete measurement data")
    for phase in phases:
        phase_rows = [row for row in rows if row["phase"] == phase]
        if sorted(int(row["repetition_index"]) for row in phase_rows) != list(range(expected if phase in selected_phases else 0)):
            raise ValueError(f"Missing or duplicate repetition in {phase}")
        output_count = 0 if phase == "prefill" else 1
        if any(int(row["output_tokens_per_request"]) != output_count for row in phase_rows):
            raise ValueError(f"Unexpected output token count in {phase}")
        if phase == "cached_decode" and any(
            int(row["prefill_cache_length"]) != config["input_tokens_per_request"]
            or int(row["decode_cache_length"]) != config["input_tokens_per_request"] + 1
            for row in phase_rows
        ):
            raise ValueError("Cached-decode lengths do not match the resolved config")
    return phase_summary(
        phases["prefill"], phases["cached_decode"],
        config["batch_size"], config["input_tokens_per_request"],
        selected_phases,
    )
