"""Capture one diagnostic Qwen forward trace for a selected workload point."""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile, record_function

from forward_workload import load_workload, prepare_decode_inputs, workload_metadata
from dunlin.artifacts import create_run_dir, new_manifest, utc_now, write_json
from dunlin.config import LOGITS_TO_KEEP, USE_CACHE, BenchmarkConfig, resolve_config
from dunlin.device import select_device, synchronize
from dunlin.environment import collect_environment
from dunlin.source import SourceCheckout, add_source_arguments

logger = logging.getLogger(__name__)


BASELINE_WORKLOAD_KEYS = (
    "model_id", "model_revision", "tokenizer_revision", "prompt_recipe_id",
    "prompt_text_sha256", "formatted_prompt_sha256", "input_token_ids_sha256",
    "batch_size", "input_tokens_per_request", "decode_tokens_per_request",
    "resolved_device", "resolved_dtype", "attention_implementation",
    "model_class", "tokenizer_class", "compiled", "use_cache", "logits_to_keep",
)
VALIDATION_RTOL = 1e-2
VALIDATION_ATOL = 1e-2


@dataclass(frozen=True)
class CapturedForward:
    """Observed outputs collected outside the forward's capture range."""

    logits: torch.Tensor
    cache_length_before: int | None
    cache_length_after: int | None
    cache_class: str | None
    cuda_kernel_events: int | None


def validate_profile_forward(
    captured: CapturedForward, *, phase: str, batch_size: int,
    input_tokens: int, vocab_size: int, use_torch_profiler: bool,
    reference_logits: torch.Tensor,
    rtol: float = VALIDATION_RTOL, atol: float = VALIDATION_ATOL,
) -> dict[str, object]:
    """Check a completed capture and return its validation record.

    Called once, outside capture, for the selected forward and its unprofiled
    reference. The disposable capture is discarded without validation.
    """
    if phase not in ("prefill", "cached_decode"):
        raise ValueError(f"Unknown forward phase: {phase}")
    if use_torch_profiler:
        if captured.cuda_kernel_events is None:
            raise RuntimeError("Profiler returned no events")
        if captured.cuda_kernel_events == 0:
            raise RuntimeError("Profiler captured no CUDA kernel events; check CUDA tracing before interpreting this run")
    if captured.cache_length_after is None:
        raise RuntimeError("Expected past_key_values when use_cache=True")
    if phase == "cached_decode" and captured.cache_length_before != input_tokens:
        raise RuntimeError(
            f"Expected fresh prompt cache length {input_tokens}; got {captured.cache_length_before}"
        )
    expected_cache_length = input_tokens + (phase == "cached_decode")
    if captured.cache_length_after != expected_cache_length:
        raise RuntimeError(f"Expected cache length {expected_cache_length}; got {captured.cache_length_after}")
    expected_shape = (batch_size, 1, vocab_size)
    if tuple(captured.logits.shape) != expected_shape or not bool(torch.isfinite(captured.logits).all()):
        raise RuntimeError(f"Invalid profiled logits: expected finite {expected_shape}, got {tuple(captured.logits.shape)}")
    if tuple(reference_logits.shape) != expected_shape:
        raise RuntimeError(f"Invalid reference logits: expected {expected_shape}, got {tuple(reference_logits.shape)}")
    if not torch.allclose(reference_logits, captured.logits, rtol=rtol, atol=atol):
        raise RuntimeError("Profiled logits differ from an unprofiled forward on fresh inputs")
    return {
        "logits_shape": list(captured.logits.shape), "logits_finite": True,
        "profiled_matches_unprofiled": True if use_torch_profiler else None,
        "forward_matches_reference": True,
        "cache_length_before": captured.cache_length_before,
        "cache_length_after": captured.cache_length_after,
        "cache_class": captured.cache_class,
        "cuda_kernel_events": captured.cuda_kernel_events,
    }


def settings_from_baseline(
    baseline_run_dir: Path, output_dir: Path | None,
    *, debug: bool = False,
) -> tuple[BenchmarkConfig, str, dict[str, object]]:
    """Use a completed timing run's effective workload, including CLI overrides."""
    manifest = json.loads((baseline_run_dir / "manifest.json").read_text())
    baseline = json.loads((baseline_run_dir / "config.resolved.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(baseline, dict):
        raise ValueError("Baseline manifest and resolved config must be JSON objects")
    if manifest.get("source_mode", "strict") != "strict" and not debug:
        raise ValueError("A debug baseline requires --debug on the profile run")
    if (manifest.get("status") != "complete" or manifest.get("run_kind") != "benchmark"
            or manifest.get("experiment") != "batched_forward_phases"
            or baseline.get("experiment") != "batched_forward_phases"):
        raise ValueError("Baseline must be a completed batched_forward_phases run")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Baseline manifest has no run ID")
    required = (*BASELINE_WORKLOAD_KEYS, "warmup_runs_per_phase", "measured_runs_per_phase")
    missing = [key for key in required if key not in baseline]
    if missing:
        raise ValueError(f"Baseline resolved config lacks workload fields: {', '.join(missing)}")
    if not baseline["model_revision"] or not baseline["tokenizer_revision"]:
        raise ValueError("Baseline needs resolved model and tokenizer revisions")
    resolved_dtype = baseline["resolved_dtype"]
    if not isinstance(resolved_dtype, str) or not resolved_dtype.startswith("torch."):
        raise ValueError("Baseline has an invalid resolved dtype")
    source_copy = baseline_run_dir / "config.source.toml"
    settings = resolve_config(
        source_copy if source_copy.is_file() else None,
        {
            "model_id": baseline["model_id"],
            "model_revision": baseline["model_revision"],
            "tokenizer_revision": baseline["tokenizer_revision"],
            "prompt_recipe": baseline["prompt_recipe_id"],
            "context_lengths": [baseline["input_tokens_per_request"]],
            "batch_sizes": [baseline["batch_size"]],
            "device": baseline["resolved_device"],
            "dtype": resolved_dtype.removeprefix("torch."),
            "warmup_runs_per_phase": baseline["warmup_runs_per_phase"],
            "measured_runs_per_phase": baseline["measured_runs_per_phase"],
            "decode_tokens_per_request": baseline["decode_tokens_per_request"],
            "use_cache": baseline["use_cache"],
            "logits_to_keep": baseline["logits_to_keep"],
        },
        output_dir=output_dir,
    )
    return settings, run_id, baseline


def validate_baseline_workload(
    baseline: dict[str, object], profile_workload: dict[str, object],
) -> None:
    """Reject a linked baseline when the loaded workload identity differs."""
    mismatches = [
        key for key in BASELINE_WORKLOAD_KEYS
        if baseline[key] != profile_workload[key]
    ]
    if mismatches:
        raise ValueError(f"Profile workload differs from baseline: {', '.join(mismatches)}")


def capture_forward(model, inputs, device: str, phase: str, *, save_to: Path | None,
                    record_shapes: bool, use_torch_profiler: bool = True,
                    ) -> CapturedForward:
    """Capture a forward and collect observations; validation is a separate step."""
    decode_inputs = prepare_decode_inputs(model, inputs, device) if phase == "cached_decode" else None
    if decode_inputs is not None:
        before = int(decode_inputs[2].get_seq_length())
        cache_class = type(decode_inputs[2]).__name__
    else:
        before = None
        cache_class = None
    synchronize(device)
    profiler_context = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=record_shapes,
        profile_memory=False,
        with_stack=False,
    ) if use_torch_profiler else nullcontext()
    with profiler_context as profiler:
        forward_range = (record_function(f"{phase}_forward") if use_torch_profiler
                         else torch.cuda.nvtx.range(f"{phase}_forward"))
        with forward_range:
            if decode_inputs is not None:
                next_tokens, attention_mask, cache = decode_inputs
                outputs = model(
                    input_ids=next_tokens,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=USE_CACHE,
                    logits_to_keep=LOGITS_TO_KEEP,
                )
            else:
                outputs = model(**inputs, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
        with record_function("wait_for_completion") if use_torch_profiler else nullcontext():
            synchronize(device)
    cuda_events = None
    if profiler is not None:
        events = profiler.events()
        cuda_events = None if events is None else sum(event.device_type == DeviceType.CUDA for event in events)
        if save_to is not None:
            profiler.export_chrome_trace(str(save_to))
            averages = profiler.key_averages(group_by_input_shape=record_shapes)
            row_limit = -1 if record_shapes else 20
            (save_to.parent / "operator_table_device.txt").write_text(
                averages.table(sort_by="self_device_time_total", row_limit=row_limit)
            )
            (save_to.parent / "operator_table_cpu.txt").write_text(
                averages.table(sort_by="self_cpu_time_total", row_limit=row_limit)
            )
    after = None if outputs.past_key_values is None else int(outputs.past_key_values.get_seq_length())
    return CapturedForward(outputs.logits.detach().clone(), before, after, cache_class, cuda_events)


def reference_forward(model, inputs, device: str, phase: str) -> torch.Tensor:
    """Run the same forward without profiling, using a fresh decode cache."""
    if phase == "cached_decode":
        next_tokens, attention_mask, cache = prepare_decode_inputs(model, inputs, device)
        outputs = model(
            input_ids=next_tokens,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=USE_CACHE,
            logits_to_keep=LOGITS_TO_KEEP,
        )
    else:
        outputs = model(**inputs, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
    synchronize(device)
    return outputs.logits.detach().clone()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Profile one Qwen prefill or cached-decode forward.")
    parser.add_argument("--config", type=Path, help="Manual mode TOML preset (default: configs/batch_context.toml)")
    parser.add_argument("--baseline-run-dir", type=Path, help="Use and validate a completed timing run's workload")
    parser.add_argument("--phase", choices=("cached_decode", "prefill"), default="cached_decode")
    parser.add_argument("--input-tokens", type=int, help="Manual mode context length (default: 500)")
    parser.add_argument("--batch-size", type=int, help="Manual mode batch size")
    parser.add_argument("--device", help="Manual mode device (default: cuda:0)")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), help="Manual mode dtype (default: float16)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--baseline-run-id", help="Manual baseline ID label; no automatic validation")
    parser.add_argument("--record-shapes", action="store_true", help="Separate diagnostic shape pass")
    parser.add_argument("--no-torch-profiler", action="store_true",
                        help="Use NVTX forward ranges for Nsight without capturing a PyTorch trace")
    parser.add_argument("--dry-run", action="store_true")
    add_source_arguments(parser)
    args = parser.parse_args()
    if args.output_dir and args.run_dir:
        parser.error("--output-dir and --run-dir cannot be used together")
    if args.no_torch_profiler and args.record_shapes:
        parser.error("--record-shapes requires the PyTorch profiler")
    if args.baseline_run_dir and any(
        value is not None for value in (
            args.config, args.input_tokens, args.batch_size, args.device,
            args.dtype, args.baseline_run_id,
        )
    ):
        parser.error("--baseline-run-dir cannot be combined with manual workload or baseline ID flags")
    if not args.baseline_run_dir and args.batch_size is None:
        parser.error("--batch-size is required without --baseline-run-dir")
    baseline: dict[str, object] | None = None
    baseline_run_id = args.baseline_run_id
    output_dir = args.output_dir or (args.run_dir.parent if args.run_dir else None)
    try:
        if args.baseline_run_dir:
            settings, baseline_run_id, baseline = settings_from_baseline(args.baseline_run_dir, output_dir, debug=args.debug)
        else:
            settings = resolve_config(
                args.config or Path("configs/batch_context.toml"),
                {"context_lengths": [args.input_tokens if args.input_tokens is not None else 500], "batch_sizes": [args.batch_size],
                 "device": args.device or "cuda:0", "dtype": args.dtype or "float16"},
                output_dir=output_dir,
            )
        device, dtype = select_device(settings.device, settings.dtype)
        if not device.startswith("cuda"):
            raise ValueError("This study requires CUDA device tracing")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"phase": args.phase, "input_tokens": settings.context_lengths[0],
                          "batch_size": settings.batch_sizes[0], "device": device, "dtype": str(dtype),
                          "record_shapes": args.record_shapes,
                          "torch_profiler_enabled": not args.no_torch_profiler,
                          "config": str(settings.source_path) if settings.source_path else None,
                          "baseline_run_id": baseline_run_id}, indent=2))
        return

    run_dir = create_run_dir("profile_forward_phases", args.run_dir, settings.output_root, debug=args.debug)
    logger.info("Starting profile; phase=%s, run_dir: %s", args.phase, run_dir)
    manifest = new_manifest("profile_forward_phases", run_dir.name, debug=args.debug)
    manifest["run_kind"] = "profile"
    manifest["linked_baseline_run_id"] = baseline_run_id
    write_json(run_dir / "manifest.json", manifest)
    if settings.source_path is not None:
        (run_dir / "config.source.toml").write_bytes(settings.source_path.read_bytes())
    try:
        source = SourceCheckout.start(run_dir, debug=args.debug)
        logger.info("Loading %s on %s (%s); batch_size=%s, input_tokens=%s",
                    settings.model_id, device, dtype, settings.batch_sizes[0], settings.context_lengths[0])
        model, tokenizer, text, single_inputs, inputs = load_workload(settings, device, dtype)
        logger.info("Workload loaded; recording metadata")
        profile_workload = workload_metadata(settings, model, tokenizer, text, single_inputs, device, dtype)
        if baseline is not None:
            validate_baseline_workload(baseline, profile_workload)
        config: dict[str, object] = {
            "schema_version": 1,
            "experiment": "profile_forward_phases",
            **profile_workload,
            "phase": args.phase,
            "architecture": {
                "hidden_size": model.config.hidden_size,
                "intermediate_size": model.config.intermediate_size,
                "num_hidden_layers": model.config.num_hidden_layers,
                "num_attention_heads": model.config.num_attention_heads,
                "num_key_value_heads": model.config.num_key_value_heads,
                "head_dim": getattr(model.config, "head_dim", None)
                    or model.config.hidden_size // model.config.num_attention_heads,
                "vocab_size": model.config.vocab_size,
            },
            "source_config": str(settings.source_path) if settings.source_path else None,
            "baseline_run_dir": str(args.baseline_run_dir.resolve()) if args.baseline_run_dir else None,
            "output_root": str(run_dir.parent),
            "torch_profiler_enabled": not args.no_torch_profiler,
            "profiler_activities": [] if args.no_torch_profiler else ["CPU", "CUDA"],
            "profiler_record_shapes": args.record_shapes,
            "profiler_profile_memory": False,
            "profiler_with_stack": False,
            "operator_table_grouping": "key_averages by operator name; top 20 rows per sort",
            "operator_table_sort_keys": ["self_device_time_total", "self_cpu_time_total"],
            "warmup_forwards": settings.warmup_runs_per_phase,
            "disposable_captures": 0 if args.no_torch_profiler else 1,
            "saved_captures": 0 if args.no_torch_profiler else 1,
            "capture_order": (
                f"{settings.warmup_runs_per_phase} phase warm-ups, two NVTX-marked forwards, one reference forward"
                if args.no_torch_profiler else
                f"{settings.warmup_runs_per_phase} phase warm-ups, one disposable capture, one saved capture, one unprofiled correctness forward"
            ),
            "capture_scope": (
                "one model forward in NVTX range followed by synchronization; cache preparation outside range"
                if args.no_torch_profiler else
                "one model forward followed by separate wait_for_completion; cache preparation outside capture"
            ),
            "forward_range": "NVTX" if args.no_torch_profiler else "torch.profiler record_function",
            "decode_cache_policy": "fresh prompt prefill before each cached decode forward",
            "validation_rtol": VALIDATION_RTOL,
            "validation_atol": VALIDATION_ATOL,
        }
        write_json(run_dir / "config.resolved.json", config)
        write_json(run_dir / "environment.json", collect_environment(device, dtype))
        source.check("before warmup")
        with torch.inference_mode():
            logger.info("%s: warming up (%s runs)", args.phase, settings.warmup_runs_per_phase)
            for _ in range(settings.warmup_runs_per_phase):
                if args.phase == "cached_decode":
                    next_tokens, attention_mask, cache = prepare_decode_inputs(model, inputs, device)
                    synchronize(device)
                    model(input_ids=next_tokens, attention_mask=attention_mask,
                          past_key_values=cache, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
                else:
                    synchronize(device)
                    model(**inputs, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
                synchronize(device)
            source.check("before initial capture")
            logger.info("%s: running initial capture", args.phase)
            capture_forward(model, inputs, device, args.phase, save_to=None,
                            record_shapes=args.record_shapes,
                            use_torch_profiler=not args.no_torch_profiler)
            source.check("before diagnostic capture")
            logger.info("%s: running diagnostic capture", args.phase)
            captured = capture_forward(
                model, inputs, device, args.phase,
                save_to=None if args.no_torch_profiler else run_dir / "trace.json",
                record_shapes=args.record_shapes,
                use_torch_profiler=not args.no_torch_profiler,
            )
            source.check("after diagnostic capture")
            logger.info("Capture complete; running reference forward and validating")
            reference_logits = reference_forward(model, inputs, device, args.phase)
            validation = validate_profile_forward(
                captured, reference_logits=reference_logits, phase=args.phase,
                batch_size=settings.batch_sizes[0], input_tokens=settings.context_lengths[0],
                vocab_size=model.config.vocab_size, use_torch_profiler=not args.no_torch_profiler,
            )
        write_json(run_dir / "validation.json", validation)
        manifest["artifacts"] = [
            "config.resolved.json", "environment.json", "validation.json", "source-checks.jsonl",
        ]
        if not args.no_torch_profiler:
            manifest["artifacts"].extend([
                "trace.json", "operator_table_device.txt", "operator_table_cpu.txt",
            ])
        if settings.source_path is not None:
            manifest["artifacts"].append("config.source.toml")
        source.check("completion")
        manifest["status"] = "complete"
        logger.info("Profile complete; artifacts written to %s", run_dir)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at_utc"] = utc_now()
        write_json(run_dir / "manifest.json", manifest)
        print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
