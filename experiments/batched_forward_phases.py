"""Measure prefill and one-token cached-decode latency for a static batch."""

from __future__ import annotations

import argparse
import io
import json
import logging
import platform
import statistics
import time
from contextlib import redirect_stdout
from pathlib import Path

import torch
import transformers

from dunlin.workload_helpers import DEFAULT_INPUT_TOKENS
from forward_workload import load_workload, prepare_decode_inputs, workload_metadata
from dunlin.artifacts import (
    FORWARD_METRIC_DEFINITIONS,
    FORWARD_METRIC_DEFINITIONS_VERSION,
    create_run_dir,
    new_manifest,
    phase_summary,
    utc_now,
    write_json,
    write_measurements,
)
from dunlin.config import (
    DECODE_TOKENS_PER_REQUEST,
    LOGITS_TO_KEEP,
    PHASES,
    USE_CACHE,
    BenchmarkConfig,
    describe_config,
    resolve_config,
)
from dunlin.device import select_device, synchronize
from dunlin.environment import collect_environment
from dunlin.source import SourceCheckout, add_source_arguments

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Measure batched Qwen prefill and cached-decode forwards."
    )
    parser.add_argument("--phase", choices=("prefill", "cached_decode", "both"), default="both",
                        help="Phase to measure (default: both)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--config", type=Path, help="Versioned TOML experiment preset")
    parser.add_argument("--output-dir", type=Path, help="Root for new run directories")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without loading a model")
    parser.add_argument("--run-dir", type=Path, help="Directory for this run's artifacts")
    parser.add_argument("--device", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--model-revision", help="40-character Git commit hash for model weights")
    parser.add_argument("--tokenizer-revision", help="40-character Git commit hash for tokenizer")
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--measured-runs", type=int)
    add_source_arguments(parser)
    args = parser.parse_args()
    if args.batch_size is None and args.config is None:
        parser.error("--batch-size is required without --config")
    if args.run_dir is not None and args.output_dir is not None:
        parser.error("--run-dir and --output-dir cannot be used together")
    try:
        settings = resolve_config(
            args.config,
            {
                "context_lengths": [args.input_tokens] if args.input_tokens is not None else None,
                "batch_sizes": [args.batch_size] if args.batch_size is not None else None,
                "device": args.device,
                "dtype": args.dtype,
                "model_revision": args.model_revision,
                "tokenizer_revision": args.tokenizer_revision,
                "warmup_runs_per_phase": args.warmup_runs,
                "measured_runs_per_phase": args.measured_runs,
            },
            default_context_lengths=(DEFAULT_INPUT_TOKENS,),
            default_batch_sizes=(1,),
            output_dir=args.output_dir or (args.run_dir.parent if args.run_dir else None),
        )
        if len(settings.context_lengths) != 1 or len(settings.batch_sizes) != 1:
            raise ValueError("A forward-phase run needs one context length and one batch size")
        device, dtype = select_device(settings.device, settings.dtype)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    phases = PHASES if args.phase == "both" else (args.phase,)
    if args.dry_run:
        preview = describe_config(settings, "batched_forward_phases", device, dtype)
        preview["phases"] = list(phases)
        if args.run_dir is not None:
            preview["run_dir"] = str(args.run_dir.resolve())
        print(json.dumps(preview, indent=2))
        return

    run_dir = create_run_dir("batched_forward_phases", args.run_dir, settings.output_root, debug=args.debug)
    logger.info("Starting benchmark; run_dir: %s", run_dir)
    manifest = new_manifest("batched_forward_phases", run_dir.name, debug=args.debug)
    write_json(run_dir / "manifest.json", manifest)
    if settings.source_path is not None:
        (run_dir / "config.source.toml").write_bytes(settings.source_path.read_bytes())
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            run_experiment(settings, run_dir, manifest, phases, debug=args.debug)
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
    artifacts = [
        "config.resolved.json", "environment.json", "measurements.csv",
        "summary.json", "stdout.log", "source-checks.jsonl",
    ]
    if settings.source_path is not None:
        artifacts.append("config.source.toml")
    manifest["artifacts"] = artifacts
    write_json(run_dir / "manifest.json", manifest)
    print(output.getvalue(), end="")
    print(f"run_dir: {run_dir}")


def run_experiment(
    settings: BenchmarkConfig, run_dir: Path, manifest: dict[str, object],
    phases: tuple[str, ...] = PHASES, *, debug: bool = False,
) -> None:
    if phases not in (("prefill",), ("cached_decode",), PHASES):
        raise ValueError("Invalid selected phases")
    source = SourceCheckout.start(run_dir, debug=debug)
    device, dtype = select_device(settings.device, settings.dtype)
    batch_size = settings.batch_sizes[0]
    input_tokens = settings.context_lengths[0]
    logger.info("Loading %s on %s (%s); batch_size=%s, input_tokens=%s",
                settings.model_id, device, dtype, batch_size, input_tokens)
    model, tokenizer, text, single_inputs, inputs = load_workload(settings, device, dtype)
    logger.info("Workload loaded; recording metadata")
    config: dict[str, object] = {
        "schema_version": 1,
        "experiment": "batched_forward_phases",
        **workload_metadata(settings, model, tokenizer, text, single_inputs, device, dtype),
        "warmup_runs_per_phase": settings.warmup_runs_per_phase,
        "measured_runs_per_phase": settings.measured_runs_per_phase,
        "phases": list(phases),
        "output_root": str(run_dir.parent),
        "source_config": str(settings.source_path) if settings.source_path else None,
        "metric_definitions_version": FORWARD_METRIC_DEFINITIONS_VERSION,
        "metric_definitions": FORWARD_METRIC_DEFINITIONS,
        "decode_cache_policy": "fresh prompt prefill before each timed decode forward",
        "timing_scope": "synchronized model forwards; loading, tokenization, transfer, and decode preparation excluded",
    }
    write_json(run_dir / "config.resolved.json", config)
    write_json(run_dir / "environment.json", collect_environment(device, dtype))
    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    prefill_cache_lengths: list[int] = []
    decode_cache_lengths: list[int] = []
    source.check("before warmup")
    with torch.inference_mode():
        if "prefill" in phases:
            logger.info("prefill: warming up (%s runs)", settings.warmup_runs_per_phase)
            for _ in range(settings.warmup_runs_per_phase):
                synchronize(device)
                model(**inputs, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
                synchronize(device)

            source.check("before prefill measurements")
            logger.info("prefill: measuring (%s runs)", settings.measured_runs_per_phase)
            for _ in range(settings.measured_runs_per_phase):
                synchronize(device)
                start = time.perf_counter()
                model(**inputs, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
                synchronize(device)
                prefill_latencies.append(time.perf_counter() - start)

        if "cached_decode" in phases:
            source.check("before cached decode warmup")
            logger.info("cached_decode: warming up (%s runs)", settings.warmup_runs_per_phase)
            for _ in range(settings.warmup_runs_per_phase):
                next_tokens, decode_attention_mask, cache = prepare_decode_inputs(model, inputs, device)
                synchronize(device)
                model(
                    input_ids=next_tokens,
                    attention_mask=decode_attention_mask,
                    past_key_values=cache,
                    use_cache=USE_CACHE,
                    logits_to_keep=LOGITS_TO_KEEP,
                )
                synchronize(device)

            source.check("before cached decode measurements")
            logger.info("cached_decode: measuring (%s runs); each sample prepares a fresh prompt cache",
                        settings.measured_runs_per_phase)
            for _ in range(settings.measured_runs_per_phase):
                next_tokens, decode_attention_mask, cache = prepare_decode_inputs(model, inputs, device)
                prefill_cache_lengths.append(int(cache.get_seq_length()))
                synchronize(device)
                start = time.perf_counter()
                decode_outputs = model(
                    input_ids=next_tokens,
                    attention_mask=decode_attention_mask,
                    past_key_values=cache,
                    use_cache=USE_CACHE,
                    logits_to_keep=LOGITS_TO_KEEP,
                )
                synchronize(device)
                decode_latencies.append(time.perf_counter() - start)
                if decode_outputs.past_key_values is None:
                    raise RuntimeError("Expected decode to return past_key_values.")
                decode_cache_lengths.append(
                    int(decode_outputs.past_key_values.get_seq_length())
                )

    logger.info("Measurements complete; validating and writing artifacts")
    if "cached_decode" in phases:
        if set(prefill_cache_lengths) != {input_tokens}:
            raise RuntimeError(f"Unexpected prefill cache lengths: {prefill_cache_lengths}")
        if set(decode_cache_lengths) != {input_tokens + DECODE_TOKENS_PER_REQUEST}:
            raise RuntimeError(f"Unexpected decode cache lengths: {decode_cache_lengths}")

    write_measurements(
        run_dir / "measurements.csv", str(manifest["run_id"]),
        batch_size, input_tokens, prefill_latencies, decode_latencies,
        prefill_cache_lengths, decode_cache_lengths,
    )
    write_json(
        run_dir / "summary.json",
        phase_summary(prefill_latencies, decode_latencies, batch_size, input_tokens, phases),
    )

    print(f"model_name: {settings.model_id}")
    print(f"pytorch_version: {torch.__version__}")
    print(f"transformers_version: {transformers.__version__}")
    print(f"platform: {platform.platform()}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"batch_size: {batch_size}")
    print(f"input_tokens_per_request: {input_tokens}")
    print(f"decode_tokens_per_request: {DECODE_TOKENS_PER_REQUEST}")
    print(f"use_cache: {USE_CACHE}")
    print(f"logits_to_keep: {LOGITS_TO_KEEP}")
    print(f"warmup_runs_per_phase: {settings.warmup_runs_per_phase}")
    print(f"measured_runs_per_phase: {settings.measured_runs_per_phase}")
    print(f"phases: {list(phases)}")
    if "prefill" in phases:
        print(f"prefill_forward_latencies_seconds: {prefill_latencies}")
        print(f"median_prefill_forward_latency_seconds: {statistics.median(prefill_latencies)}")
    if "cached_decode" in phases:
        print(f"cached_decode_forward_latencies_seconds: {decode_latencies}")
        print(f"median_cached_decode_forward_latency_seconds: {statistics.median(decode_latencies)}")
    print(
        "timing_scope: synchronized model(...) calls; prefill includes the full-prompt "
        "transformer forward, final-position projection, and cache construction; "
        "cached decode includes only the one-token forward against a fresh cache"
    )

    source.check("completion")


if __name__ == "__main__":
    main()
