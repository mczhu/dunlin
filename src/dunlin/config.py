"""Resolve presets and runtime settings for forward-phase runs and matrices."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from dunlin.device import DTYPES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_WARMUP_RUNS = 5
DEFAULT_MEASURED_RUNS = 30
DEFAULT_CONTEXT_LENGTHS = (32, 64, 128, 250, 500)
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
PROMPT_RECIPE = "city_planning_qwen_chat_v1"
PHASES = ("prefill", "cached_decode")
DECODE_TOKENS_PER_REQUEST = 1
USE_CACHE = True
LOGITS_TO_KEEP = 1
PRESET_KEYS = {
    "schema_version", "model_id", "model_revision", "tokenizer_revision",
    "prompt_recipe", "context_lengths", "batch_sizes", "device", "dtype",
    "warmup_runs_per_phase", "measured_runs_per_phase", "phases",
    "decode_tokens_per_request", "use_cache", "logits_to_keep",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str
    model_revision: str | None
    tokenizer_revision: str | None
    prompt_recipe: str
    context_lengths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    device: str
    dtype: str
    warmup_runs_per_phase: int
    measured_runs_per_phase: int
    source_path: Path | None
    output_root: Path


def describe_config(
    config: BenchmarkConfig, experiment: str, resolved_device: str, resolved_dtype: object,
) -> dict[str, object]:
    """Show the effective workload without loading a model."""
    description: dict[str, object] = {
        "experiment": experiment,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "prompt_recipe": config.prompt_recipe,
        "context_lengths": list(config.context_lengths),
        "batch_sizes": list(config.batch_sizes),
        "requested_device": config.device,
        "resolved_device": resolved_device,
        "requested_dtype": config.dtype,
        "resolved_dtype": str(resolved_dtype),
        "warmup_runs_per_phase": config.warmup_runs_per_phase,
        "measured_runs_per_phase": config.measured_runs_per_phase,
        "phases": list(PHASES),
        "decode_tokens_per_request": DECODE_TOKENS_PER_REQUEST,
        "use_cache": USE_CACHE,
        "logits_to_keep": LOGITS_TO_KEEP,
        "output_root": str(config.output_root),
        "source_config": str(config.source_path) if config.source_path else None,
    }
    if experiment == "batch_context_matrix":
        description["matrix_points"] = [
            {"context_length": context, "batch_size": batch}
            for context in config.context_lengths for batch in config.batch_sizes
        ]
    return description


def resolve_output_root(output_dir: Path | None, environ: Mapping[str, str] = os.environ) -> Path:
    if output_dir is not None:
        chosen = output_dir
    elif environ.get("DUNLIN_RESULTS"):
        chosen = Path(environ["DUNLIN_RESULTS"])
    elif environ.get("INFERENCE_LAB_RESULTS"):
        chosen = Path(environ["INFERENCE_LAB_RESULTS"])
    else:
        chosen = ROOT / "runs"
    return chosen.resolve()


def _ordered_positive_ints(value: object, key: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{key} must be a nonempty ordered list")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{key} must contain positive integers")
    result = cast(tuple[int, ...], tuple(value))
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{key} must be strictly increasing without duplicates")
    return result


def resolve_config(
    source_path: Path | None,
    overrides: Mapping[str, object],
    *,
    default_context_lengths: tuple[int, ...] = DEFAULT_CONTEXT_LENGTHS,
    default_batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    output_dir: Path | None = None,
    environ: Mapping[str, str] = os.environ,
) -> BenchmarkConfig:
    """Apply CLI > TOML > current defaults, then validate the effective workload."""
    values: dict[str, Any] = {}
    if source_path is not None:
        source_path = source_path.resolve()
        try:
            with source_path.open("rb") as source:
                values = tomllib.load(source)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"Invalid TOML preset: {error}") from error
        unknown = values.keys() - PRESET_KEYS
        if unknown:
            raise ValueError(f"Unknown preset keys: {', '.join(sorted(unknown))}")
        if type(values.get("schema_version")) is not int or values["schema_version"] != 1:
            raise ValueError("Preset schema_version must be 1")

    defaults: dict[str, object] = {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": None,
        "tokenizer_revision": None,
        "prompt_recipe": PROMPT_RECIPE,
        "context_lengths": default_context_lengths,
        "batch_sizes": default_batch_sizes,
        "device": "auto",
        "dtype": "auto",
        "warmup_runs_per_phase": DEFAULT_WARMUP_RUNS,
        "measured_runs_per_phase": DEFAULT_MEASURED_RUNS,
        "phases": PHASES,
        "decode_tokens_per_request": DECODE_TOKENS_PER_REQUEST,
        "use_cache": USE_CACHE,
        "logits_to_keep": LOGITS_TO_KEEP,
    }
    defaults.update(values)
    unknown_overrides = overrides.keys() - PRESET_KEYS
    if unknown_overrides:
        raise ValueError(f"Unknown overrides: {', '.join(sorted(unknown_overrides))}")
    defaults.update({key: value for key, value in overrides.items() if value is not None})

    model_id = defaults["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    for key in ("model_revision", "tokenizer_revision"):
        revision = defaults[key]
        if revision is not None and (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise ValueError(f"{key} must be a 40-character commit hash")
    if defaults["prompt_recipe"] != PROMPT_RECIPE:
        raise ValueError(f"Unsupported prompt_recipe: {defaults['prompt_recipe']}")
    context_lengths = _ordered_positive_ints(defaults["context_lengths"], "context_lengths")
    batch_sizes = _ordered_positive_ints(defaults["batch_sizes"], "batch_sizes")
    device = defaults["device"]
    dtype = defaults["dtype"]
    if not isinstance(device, str) or not (
        device in ("auto", "cpu", "mps", "cuda")
        or re.fullmatch(r"cuda:[0-9]+", device)
    ):
        raise ValueError(f"Invalid device: {device}")
    if not isinstance(dtype, str) or dtype not in ("auto", *DTYPES):
        raise ValueError(f"Invalid dtype: {dtype}")
    for key, minimum in (("warmup_runs_per_phase", 0), ("measured_runs_per_phase", 1)):
        value = defaults[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    phases = defaults["phases"]
    if not isinstance(phases, (list, tuple)) or tuple(phases) != PHASES:
        raise ValueError("phases must be ['prefill', 'cached_decode'] in that order")
    if (
        type(defaults["decode_tokens_per_request"]) is not int
        or defaults["decode_tokens_per_request"] != DECODE_TOKENS_PER_REQUEST
    ):
        raise ValueError(f"decode_tokens_per_request must be {DECODE_TOKENS_PER_REQUEST}")
    if (
        defaults["use_cache"] is not USE_CACHE
        or type(defaults["logits_to_keep"]) is not int
        or defaults["logits_to_keep"] != LOGITS_TO_KEEP
    ):
        raise ValueError(
            f"use_cache={str(USE_CACHE).lower()} and logits_to_keep={LOGITS_TO_KEEP} are required"
        )

    return BenchmarkConfig(
        model_id=model_id,
        model_revision=cast(str | None, defaults["model_revision"]),
        tokenizer_revision=cast(str | None, defaults["tokenizer_revision"]),
        prompt_recipe=PROMPT_RECIPE,
        context_lengths=context_lengths,
        batch_sizes=batch_sizes,
        device=device,
        dtype=dtype,
        warmup_runs_per_phase=cast(int, defaults["warmup_runs_per_phase"]),
        measured_runs_per_phase=cast(int, defaults["measured_runs_per_phase"]),
        source_path=source_path,
        output_root=resolve_output_root(output_dir, environ),
    )


def resolve_revisions(
    model_id: str,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
) -> tuple[str, str]:
    """Resolve model and tokenizer Git commit hashes to ensure immutable provenance.

    When benchmarking without an explicit preset, revisions default to None, leaving
    the target repository pointing to a mutable branch ref (such as 'main'). Resolving
    the underlying 40-character commit hash upfront guarantees that:
    1. The parent orchestrator and all child workers record the exact same commit hash.
    2. Long-running or multi-process matrix benchmarks are protected against upstream
       repository updates during the run.
    3. Final matrix validation can verify identical model artifacts across all points.
    """
    if model_revision is not None and tokenizer_revision is not None:
        return model_revision, tokenizer_revision

    from transformers import AutoConfig

    config_kwargs = {"revision": model_revision} if model_revision else {}
    config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    resolved_commit = getattr(config, "_commit_hash", None)
    if not isinstance(resolved_commit, str) or re.fullmatch(r"[0-9a-f]{40}", resolved_commit) is None:
        raise ValueError(f"Could not resolve 40-character commit hash for model {model_id}")

    return model_revision or resolved_commit, tokenizer_revision or resolved_commit

