"""Validate and summarize saved context-by-batch matrix runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dunlin.artifacts import summarize_saved_run

MATRIX_SUMMARY_SCHEMA_VERSION = 2
MATRIX_METRIC_DEFINITIONS_VERSION = "batch_context_matrix_metrics_v1"
MATRIX_METRIC_DEFINITIONS = {
    "median_prefill_forward_latency_ms": {
        "formula": "child median_prefill_forward_latency_seconds * 1000",
        "unit": "milliseconds",
    },
    "median_cached_decode_forward_latency_ms": {
        "formula": "child median_cached_decode_forward_latency_seconds * 1000",
        "unit": "milliseconds",
    },
    "prefill_throughput_speedup_vs_batch_one": {
        "formula": "batch_size * batch_one_prefill_latency / this_batch_prefill_latency at the same context length",
        "unit": "ratio",
    },
    "decode_throughput_speedup_vs_batch_one": {
        "formula": "batch_size * batch_one_decode_latency / this_batch_decode_latency at the same context length",
        "unit": "ratio",
    },
}


@dataclass(frozen=True)
class MatrixRun:
    path: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    environment: dict[str, Any]
    child_configs: dict[tuple[int, int], dict[str, Any]]
    child_methodology_versions: dict[tuple[int, int], str]
    summary: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_matrix_run(run_dir: Path, *, allow_running: bool = False) -> MatrixRun:
    """Rebuild a complete matrix from child samples, ignoring saved medians."""
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("source_mode", "strict") not in ("strict", "debug"):
        raise ValueError("Unknown matrix source mode")
    config = _read_json(run_dir / "config.resolved.json")
    environment = _read_json(run_dir / "environment.json")
    if manifest.get("run_kind") != "benchmark" or manifest.get("experiment") != "batch_context_matrix":
        raise ValueError("Expected a matrix benchmark run")
    if not isinstance(manifest.get("methodology_version"), str):
        raise ValueError("Matrix methodology version is missing")
    allowed_statuses = ("complete", "running") if allow_running else ("complete",)
    if manifest.get("status") not in allowed_statuses:
        raise ValueError("Only completed matrices can be plotted or compared")
    contexts = config.get("context_lengths")
    batches = config.get("batch_sizes")
    if not isinstance(contexts, list) or not isinstance(batches, list) or not contexts or not batches:
        raise ValueError("Matrix config must list context lengths and batch sizes")
    if (
        any(type(value) is not int or value <= 0 for value in contexts + batches)
        or batches[0] != 1
        or len(set(contexts)) != len(contexts)
        or len(set(batches)) != len(batches)
    ):
        raise ValueError("Matrix shapes must be unique and include batch one first")
    if config.get("metric_definitions_version") not in (None, MATRIX_METRIC_DEFINITIONS_VERSION):
        raise ValueError("Unsupported matrix metric definitions version")
    if "metric_definitions" in config and config["metric_definitions"] != MATRIX_METRIC_DEFINITIONS:
        raise ValueError("Matrix metric definitions differ from the supported definitions")
    expected = [(context, batch) for context in contexts for batch in batches]
    entries = manifest.get("matrix_points", manifest.get("points"))
    if not isinstance(entries, list) or [
        (entry.get("context_length"), entry.get("batch_size")) for entry in entries
    ] != expected:
        raise ValueError("Matrix manifest has missing, duplicate, or reordered points")

    child_configs: dict[tuple[int, int], dict[str, Any]] = {}
    child_methodology_versions: dict[tuple[int, int], str] = {}
    child_ids: list[str] = []
    medians: dict[tuple[int, int], tuple[float, float]] = {}
    input_hashes: dict[int, str] = {}
    child_methodology: str | None = None
    for entry in entries:
        if entry.get("status") != "complete":
            raise ValueError(f"Matrix point is incomplete: {entry}")
        child_id = entry.get("run_id")
        if not isinstance(child_id, str) or Path(child_id).name != child_id or child_id in child_ids:
            raise ValueError("Matrix child run IDs must be unique directory names")
        child_ids.append(child_id)
        child_dir = run_dir / child_id
        child_manifest = _read_json(child_dir / "manifest.json")
        if child_manifest.get("source_mode", "strict") != manifest.get("source_mode", "strict"):
            raise ValueError(f"Child {child_id} source mode differs from matrix")
        if child_manifest.get("run_id") != child_id or child_manifest.get("experiment") != "batched_forward_phases":
            raise ValueError(f"Child manifest does not match {child_id}")
        if not isinstance(child_manifest.get("methodology_version"), str):
            raise ValueError(f"Child methodology version is missing: {child_id}")
        if child_methodology is not None and child_manifest["methodology_version"] != child_methodology:
            raise ValueError("Matrix children use different methodology versions")
        child_methodology = child_manifest["methodology_version"]
        child_config = _read_json(child_dir / "config.resolved.json")
        child_environment = _read_json(child_dir / "environment.json")
        shape = (entry["context_length"], entry["batch_size"])
        if (
            child_config.get("input_tokens_per_request") != shape[0]
            or child_config.get("batch_size") != shape[1]
        ):
            raise ValueError(f"Child workload does not match matrix point {child_id}")
        for key in (
            "model_id", "model_revision", "tokenizer_revision", "prompt_recipe_id",
            "measured_runs_per_phase", "warmup_runs_per_phase", "resolved_dtype",
            "resolved_device", "decode_tokens_per_request", "use_cache", "logits_to_keep",
        ):
            if key in config and child_config.get(key) != config[key]:
                raise ValueError(f"Child {child_id} differs from matrix config: {key}")
        for key in (
            "backend", "device", "dtype", "git_commit", "git_dirty", "uv_lock_sha256",
            "python_version", "torch_version", "transformers_version",
            "runtime_settings", "torch_cuda_build_version",
        ):
            if manifest.get("source_mode") == "debug" and key in ("git_commit", "git_dirty", "uv_lock_sha256"):
                continue
            if key in environment and child_environment.get(key) != environment[key]:
                raise ValueError(f"Child {child_id} differs from matrix environment: {key}")
        input_hash = child_config.get("input_token_ids_sha256")
        if input_hash is not None:
            if shape[0] in input_hashes and input_hashes[shape[0]] != input_hash:
                raise ValueError(f"Input token IDs differ across batches at context {shape[0]}")
            input_hashes[shape[0]] = input_hash
        child_summary = summarize_saved_run(child_dir)
        medians[shape] = (
            child_summary["median_prefill_forward_latency_seconds"] * 1000,
            child_summary["median_cached_decode_forward_latency_seconds"] * 1000,
        )
        child_configs[shape] = child_config
        child_methodology_versions[shape] = child_manifest["methodology_version"]

    points: list[dict[str, float | int]] = []
    for context, batch in expected:
        prefill_ms, decode_ms = medians[context, batch]
        batch_one_prefill, batch_one_decode = medians[context, 1]
        points.append({
            "context_length": context,
            "batch_size": batch,
            "median_prefill_forward_latency_ms": prefill_ms,
            "median_cached_decode_forward_latency_ms": decode_ms,
            "prefill_throughput_speedup_vs_batch_one": batch * batch_one_prefill / prefill_ms,
            "decode_throughput_speedup_vs_batch_one": batch * batch_one_decode / decode_ms,
        })
    summary = {
        "schema_version": MATRIX_SUMMARY_SCHEMA_VERSION,
        "source_run_ids": child_ids,
        "matrix_points": points,
    }
    if manifest.get("source_mode") == "debug":
        summary.update(source_mode="debug", publication_eligible=False)
    return MatrixRun(
        run_dir, manifest, config, environment, child_configs,
        child_methodology_versions, summary,
    )


def compare_matrix_runs(
    baseline: MatrixRun, candidate: MatrixRun, *, allow_cross_machine: bool = False,
    allow_debug: bool = False,
) -> dict[str, Any]:
    """Require matching workloads; explicitly allow labeled exploratory debug comparisons."""
    debug = any(run.manifest.get("source_mode") == "debug" for run in (baseline, candidate))
    if debug and not allow_debug:
        raise ValueError("Debug runs cannot be used for matched matrix comparisons")
    mismatches: list[str] = []
    parent_keys = (
        "model_id", "model_revision", "tokenizer_revision", "prompt_recipe_id",
        "context_lengths", "batch_sizes", "resolved_dtype", "warmup_runs_per_phase",
        "measured_runs_per_phase", "phases", "decode_tokens_per_request",
        "use_cache", "logits_to_keep", "point_process_policy", "metric_definitions_version",
        "metric_definitions",
    )
    child_keys = (
        "model_class", "tokenizer_class", "model_revision", "tokenizer_revision",
        "prompt_recipe_id", "prompt_text_sha256", "formatted_prompt_sha256",
        "input_token_ids_sha256", "attention_implementation", "compiled",
        "decode_cache_policy", "timing_scope", "use_cache", "logits_to_keep",
        "decode_tokens_per_request", "metric_definitions_version",
    )

    def check(label: str, left: dict[str, Any], right: dict[str, Any], keys: tuple[str, ...]) -> None:
        for key in keys:
            if left.get(key) is None or right.get(key) is None:
                mismatches.append(f"{label}.{key}: missing provenance")
            elif left[key] != right[key]:
                mismatches.append(f"{label}.{key}: {left[key]!r} != {right[key]!r}")

    check("matrix", baseline.config, candidate.config, parent_keys)
    if baseline.manifest.get("methodology_version") != candidate.manifest.get("methodology_version"):
        mismatches.append("matrix methodology_version differs")
    for shape in baseline.child_configs.keys() & candidate.child_configs.keys():
        check(str(shape), baseline.child_configs[shape], candidate.child_configs[shape], child_keys)
        if baseline.child_methodology_versions[shape] != candidate.child_methodology_versions[shape]:
            mismatches.append(f"{shape}.methodology_version differs")
    if baseline.child_configs.keys() != candidate.child_configs.keys():
        mismatches.append("matrix point shapes differ")
    for key in ("git_commit", "uv_lock_sha256", "python_version", "torch_version", "transformers_version"):
        if debug and key in ("git_commit", "uv_lock_sha256"):
            continue
        if baseline.environment.get(key) is None or candidate.environment.get(key) is None:
            mismatches.append(f"environment.{key}: missing provenance")
        elif baseline.environment[key] != candidate.environment[key]:
            mismatches.append(f"environment.{key} differs")
    if not debug and (baseline.environment.get("git_dirty") is not False
                      or candidate.environment.get("git_dirty") is not False):
        mismatches.append("environment.git_dirty: clean source required for exact comparison")

    machine_keys = (
        "backend", "device", "device_name", "platform", "architecture", "os_release",
        "wsl", "cpu_processor", "system_physical_memory_bytes",
        "dedicated_device_memory_bytes", "shared_system_memory_bytes",
        "torch_cuda_build_version", "nvidia_driver_version", "runtime_settings",
    )
    machine_differences = {
        key: {"baseline": baseline.environment.get(key), "candidate": candidate.environment.get(key)}
        for key in machine_keys
        if baseline.environment.get(key) != candidate.environment.get(key)
    }
    if machine_differences and not allow_cross_machine:
        mismatches.append("machine/backend differs; pass --allow-cross-machine to compare explicitly")
    if mismatches:
        raise ValueError("Incompatible matrix runs: " + "; ".join(mismatches))
    comparison = {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "compatible": True,
        "cross_machine": bool(machine_differences),
        "machine_differences": machine_differences,
    }
    if debug:
        comparison.update({
            "source_mode": "debug", "publication_eligible": False,
            "source_provenance": [{
                "run_id": run.run_id, "source_mode": run.manifest.get("source_mode", "strict"),
                "git_commit": run.environment.get("git_commit"),
                "git_dirty": run.environment.get("git_dirty"),
                "uv_lock_sha256": run.environment.get("uv_lock_sha256"),
            } for run in (baseline, candidate)],
        })
    return comparison
