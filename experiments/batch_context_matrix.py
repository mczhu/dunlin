"""Measure batching effects across prompt and batch sizes for prefill and decode."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import subprocess
import sys
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable

from dunlin.artifacts import (
    create_run_dir,
    new_manifest,
    summarize_saved_run,
    utc_now,
    write_json,
)
from dunlin.config import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_CONTEXT_LENGTHS,
    DECODE_TOKENS_PER_REQUEST,
    LOGITS_TO_KEEP,
    PHASES,
    USE_CACHE,
    BenchmarkConfig,
    describe_config,
    resolve_config,
    resolve_revisions,
)
from dunlin.device import select_device
from dunlin.environment import collect_environment
from dunlin.matrix_results import (
    MATRIX_METRIC_DEFINITIONS,
    MATRIX_METRIC_DEFINITIONS_VERSION,
    load_matrix_run,
)

from dunlin.source import SourceCheckout, add_source_arguments

logger = logging.getLogger(__name__)


CONTEXT_LENGTHS = DEFAULT_CONTEXT_LENGTHS
BATCH_SIZES = DEFAULT_BATCH_SIZES
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "experiments" / "batched_forward_phases.py"
MATRIX_MANIFEST_SCHEMA_VERSION = 2


def run_benchmark(
    context_length: int, batch_size: int, run_dir: Path | None = None,
    requested_device: str = "auto", requested_dtype: str = "auto",
    settings: BenchmarkConfig | None = None, *, debug: bool = False,
) -> tuple[float, float]:
    """Execute a single-point forward benchmark in an isolated subprocess.

    Spawns `batched_forward_phases.py` with the requested shape and settings.
    If `settings` provides pinned model/tokenizer revisions, passes them via CLI
    flags so the child records the exact commit hash resolved by the parent.

    Returns median prefill and cached-decode latency in milliseconds.
    """
    if run_dir is None:
        run_dir = ROOT / "runs" / f"n{context_length}_b{batch_size}_{uuid.uuid4().hex[:8]}"
    command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--input-tokens",
            str(context_length),
            "--batch-size",
            str(batch_size),
    ]
    command.extend(("--run-dir", str(run_dir)))
    if debug:
        command.append("--debug")
    if settings is not None:
        requested_device, requested_dtype = settings.device, settings.dtype
        command.extend((
            "--warmup-runs", str(settings.warmup_runs_per_phase),
            "--measured-runs", str(settings.measured_runs_per_phase),
        ))
        if settings.source_path is not None:
            command.extend(("--config", str(settings.source_path)))
        if settings.model_revision is not None:
            command.extend(("--model-revision", settings.model_revision))
        if settings.tokenizer_revision is not None:
            command.extend(("--tokenizer-revision", settings.tokenizer_revision))
    command.extend(("--device", requested_device, "--dtype", requested_dtype))
    logger.info("Starting matrix point; context_length=%s, batch_size=%s, run_dir=%s",
                context_length, batch_size, run_dir)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        if run_dir.exists():
            (run_dir / "stderr.log").write_text(completed.stderr)
        raise RuntimeError(
            f"Benchmark failed at context={context_length}, batch={batch_size}: "
            f"{completed.stderr.strip()}"
        )
    logger.info("Matrix point complete; context_length=%s, batch_size=%s",
                context_length, batch_size)
    summary = summarize_saved_run(run_dir)
    return (
        summary["median_prefill_forward_latency_seconds"] * 1000,
        summary["median_cached_decode_forward_latency_seconds"] * 1000,
    )


def throughput_speedups(
    latencies_ms: list[float], batch_sizes: tuple[int, ...] = BATCH_SIZES,
) -> list[float]:
    """Convert static-batch latencies to throughput speedup versus batch one."""
    if (
        not batch_sizes or batch_sizes[0] != 1
        or len(latencies_ms) != len(batch_sizes)
        or any(not math.isfinite(latency) or latency <= 0 for latency in latencies_ms)
    ):
        raise ValueError("Expected a positive latency for each batch size, starting at batch one")
    return [
        batch_size * latencies_ms[0] / latency
        for batch_size, latency in zip(batch_sizes, latencies_ms, strict=True)
    ]


def collect_measurements(
    benchmark: Callable[[int, int], tuple[float, float]] = run_benchmark,
    context_lengths: tuple[int, ...] = CONTEXT_LENGTHS,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Measure every configured context-length and batch-size combination."""
    measurements: dict[int, dict[int, tuple[float, float]]] = {}
    for context_length in context_lengths:
        measurements[context_length] = {
            batch_size: benchmark(context_length, batch_size)
            for batch_size in batch_sizes
        }
    return measurements


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Measure a context-length by batch-size matrix.")
    parser.add_argument("--config", type=Path, help="Versioned TOML experiment preset")
    parser.add_argument("--output-dir", type=Path, help="Root for new run directories")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without loading a model")
    parser.add_argument("--context-lengths", nargs="+", type=int)
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--device", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--model-revision", help="40-character Git commit hash for model weights")
    parser.add_argument("--tokenizer-revision", help="40-character Git commit hash for tokenizer")
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--measured-runs", type=int)
    add_source_arguments(parser)
    args = parser.parse_args()
    try:
        settings = resolve_config(
            args.config,
            {
                "context_lengths": args.context_lengths,
                "batch_sizes": args.batch_sizes,
                "device": args.device,
                "dtype": args.dtype,
                "model_revision": args.model_revision,
                "tokenizer_revision": args.tokenizer_revision,
                "warmup_runs_per_phase": args.warmup_runs,
                "measured_runs_per_phase": args.measured_runs,
            },
            output_dir=args.output_dir,
        )
        if settings.batch_sizes[0] != 1:
            raise ValueError("batch_sizes must start with 1 for speedup versus batch one")
        device, dtype = select_device(settings.device, settings.dtype)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(describe_config(settings, "batch_context_matrix", device, dtype), indent=2))
        return
    run_dir = create_run_dir("batch_context_matrix", output_root=settings.output_root, debug=args.debug)
    logger.info("Starting matrix; run_dir: %s", run_dir)
    manifest = new_manifest("batch_context_matrix", run_dir.name, debug=args.debug)
    manifest["schema_version"] = MATRIX_MANIFEST_SCHEMA_VERSION
    manifest["matrix_points"] = [
        {"context_length": context, "batch_size": batch, "run_id": f"n{context}_b{batch}", "status": "pending"}
        for context in settings.context_lengths for batch in settings.batch_sizes
    ]
    write_json(run_dir / "manifest.json", manifest)
    if settings.source_path is not None:
        (run_dir / "config.source.toml").write_bytes(settings.source_path.read_bytes())
    output = io.StringIO()
    try:
        source = SourceCheckout.start(run_dir, debug=args.debug)
        logger.info("Resolving model and tokenizer revisions for %s", settings.model_id)
        # Pin model/tokenizer revisions once for the parent and every child.
        model_revision, tokenizer_revision = resolve_revisions(
            settings.model_id, settings.model_revision, settings.tokenizer_revision,
        )
        settings = replace(settings, model_revision=model_revision,
                           tokenizer_revision=tokenizer_revision)
        write_json(run_dir / "config.resolved.json", {
            "schema_version": 1,
            "experiment": "batch_context_matrix",
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "tokenizer_revision": settings.tokenizer_revision,
            "prompt_recipe_id": settings.prompt_recipe,
            "context_lengths": list(settings.context_lengths),
            "batch_sizes": list(settings.batch_sizes),
            "requested_device": settings.device,
            "resolved_device": device,
            "requested_dtype": settings.dtype,
            "resolved_dtype": str(dtype),
            "warmup_runs_per_phase": settings.warmup_runs_per_phase,
            "measured_runs_per_phase": settings.measured_runs_per_phase,
            "phases": list(PHASES),
            "output_root": str(settings.output_root),
            "source_config": str(settings.source_path) if settings.source_path else None,
            "metric_definitions_version": MATRIX_METRIC_DEFINITIONS_VERSION,
            "metric_definitions": MATRIX_METRIC_DEFINITIONS,
            "decode_tokens_per_request": DECODE_TOKENS_PER_REQUEST,
            "use_cache": USE_CACHE,
            "logits_to_keep": LOGITS_TO_KEEP,
            "point_process_policy": "one model process per point in context-then-batch order",
        })
        write_json(run_dir / "environment.json", collect_environment(device, dtype))
        source.check("before matrix")
        with redirect_stdout(output):
            run_matrix(run_dir, manifest, settings, source=source)
        source.check("completion")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["finished_at_utc"] = utc_now()
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "stdout.log").write_text(output.getvalue())
        print(output.getvalue(), end="")
        raise
    (run_dir / "stdout.log").write_text(output.getvalue())
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = utc_now()
    artifacts = ["config.resolved.json", "environment.json", "summary.json", "stdout.log", "source-checks.jsonl"]
    if settings.source_path is not None:
        artifacts.append("config.source.toml")
    manifest["artifacts"] = artifacts
    write_json(run_dir / "manifest.json", manifest)
    print(output.getvalue(), end="")
    logger.info("Matrix complete; artifacts written to %s", run_dir)
    print(f"run_dir: {run_dir}")


def run_matrix(
    run_dir: Path, manifest: dict[str, object],
    settings: BenchmarkConfig, *, source: SourceCheckout | None = None,
) -> None:
    """Measure every matrix point in an isolated child process and rebuild summary.

    Each (context_length, batch_size) point is run in its own fresh process to prevent
    KV-cache or CUDA state pollution across points. After all points finish, validates
    child provenance against the parent matrix config before assembling summary.json.
    """
    source = source or SourceCheckout.start(run_dir)
    matrix_points = manifest["matrix_points"]
    assert isinstance(matrix_points, list)
    points_by_shape: dict[tuple[int, int], dict[str, object]] = {}
    for point in matrix_points:
        assert isinstance(point, dict)
        points_by_shape[point["context_length"], point["batch_size"]] = point

    def measured_point(context_length: int, batch_size: int) -> tuple[float, float]:
        point = points_by_shape[context_length, batch_size]
        point["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)
        child_dir = run_dir / str(point["run_id"])
        try:
            source.check(f"before N={context_length} B={batch_size}")
            result = run_benchmark(
                context_length, batch_size, child_dir, settings=settings, debug=source.debug,
            )
            source.check(f"after N={context_length} B={batch_size}")
        except Exception:
            point["status"] = "failed"
            write_json(run_dir / "manifest.json", manifest)
            raise
        point["status"] = "complete"
        write_json(run_dir / "manifest.json", manifest)
        return result

    measurements = collect_measurements(
        measured_point, settings.context_lengths, settings.batch_sizes,
    )

    # Rebuild the parent summary from completed child samples, not console output.
    write_json(run_dir / "summary.json", load_matrix_run(run_dir, allow_running=True).summary)

    for context_length in settings.context_lengths:
        prefill_ms = [
            measurements[context_length][batch_size][0] for batch_size in settings.batch_sizes
        ]
        decode_ms = [
            measurements[context_length][batch_size][1] for batch_size in settings.batch_sizes
        ]
        print(f"context_{context_length}_prefill_ms: {prefill_ms}")
        print(f"context_{context_length}_decode_ms: {decode_ms}")
        print(
            f"context_{context_length}_prefill_throughput_speedup: "
            f"{throughput_speedups(prefill_ms, settings.batch_sizes)}"
        )
        print(
            f"context_{context_length}_decode_throughput_speedup: "
            f"{throughput_speedups(decode_ms, settings.batch_sizes)}"
        )
    print(f"plot_command: uv run --locked python experiments/plot_batch_context_matrix.py {run_dir}")


if __name__ == "__main__":
    main()
