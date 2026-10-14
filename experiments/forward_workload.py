"""Shared Qwen workload setup for ordinary forwards and diagnostic profiling."""

from __future__ import annotations

import hashlib

import torch
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, Qwen2ForCausalLM
from transformers.cache_utils import Cache

from dunlin.workload_helpers import BENCHMARK_PROMPT, repeat_batch
from dunlin.config import DECODE_TOKENS_PER_REQUEST, LOGITS_TO_KEEP, USE_CACHE, BenchmarkConfig


def load_workload(
    settings: BenchmarkConfig, device: str, dtype: torch.dtype,
) -> tuple[Qwen2ForCausalLM, PreTrainedTokenizerBase, str, BatchEncoding, BatchEncoding]:
    """Prepare the model, chat prompt, token IDs, and batch for forward-phase runs."""
    batch_size = settings.batch_sizes[0]
    input_tokens = settings.context_lengths[0]
    tokenizer_kwargs = {"revision": settings.tokenizer_revision} if settings.tokenizer_revision else {}
    model_kwargs = {"revision": settings.model_revision} if settings.model_revision else {}
    model_config = AutoConfig.from_pretrained(settings.model_id, **model_kwargs)
    if model_config.model_type != "qwen2":
        raise ValueError(
            f"This forward-phase runner requires a Qwen2 causal LM; "
            f"{settings.model_id} declares model_type={model_config.model_type!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, **tokenizer_kwargs)
    model = Qwen2ForCausalLM.from_pretrained(
        settings.model_id,
        config=model_config,
        dtype=dtype,
        **model_kwargs,  # pyright: ignore[reportArgumentType]
    )
    loaded_model_revision = getattr(model.config, "_commit_hash", None)
    loaded_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if settings.model_revision and loaded_model_revision not in (None, settings.model_revision):
        raise RuntimeError("Loaded model revision differs from the requested revision")
    if settings.tokenizer_revision and loaded_tokenizer_revision not in (None, settings.tokenizer_revision):
        raise RuntimeError("Loaded tokenizer revision differs from the requested revision")
    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": BENCHMARK_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenizer.truncation_side = "left"
    single_inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=input_tokens,
    )
    if single_inputs["input_ids"].shape[1] != input_tokens:
        raise RuntimeError(f"Expected exactly {input_tokens} input tokens.")
    inputs = repeat_batch(single_inputs, batch_size).to(device)
    return model, tokenizer, text, single_inputs, inputs


def workload_metadata(
    settings: BenchmarkConfig, model: Qwen2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase, text: str, single_inputs: BatchEncoding,
    device: str, dtype: torch.dtype,
) -> dict[str, object]:
    """Record workload identity identically for timing and profiling runs."""
    token_ids = single_inputs["input_ids"].contiguous().numpy().tobytes()
    return {
        "model_id": settings.model_id,
        "model_revision": getattr(model.config, "_commit_hash", None) or settings.model_revision,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or settings.tokenizer_revision,
        "requested_model_revision": settings.model_revision,
        "requested_tokenizer_revision": settings.tokenizer_revision,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_dimensions": {
            **{
                key: getattr(model.config, key, None)
                for key in (
                    "num_hidden_layers", "hidden_size", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "vocab_size",
                )
            },
            "head_dim": getattr(model.config, "head_dim", None)
            or model.config.hidden_size // model.config.num_attention_heads,
        },
        # Preserve the recorded recipe description for saved-run compatibility.
        "prompt_recipe": "generate.BENCHMARK_PROMPT with Qwen user chat template, generation prompt, left truncation",
        "prompt_recipe_id": settings.prompt_recipe,
        "prompt_text_sha256": hashlib.sha256(BENCHMARK_PROMPT.encode()).hexdigest(),
        "formatted_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "input_token_ids_sha256": hashlib.sha256(token_ids).hexdigest(),
        "batch_size": settings.batch_sizes[0],
        "input_tokens_per_request": settings.context_lengths[0],
        "decode_tokens_per_request": DECODE_TOKENS_PER_REQUEST,
        "requested_device": settings.device,
        "resolved_device": device,
        "requested_dtype": settings.dtype,
        "resolved_dtype": str(dtype),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "compiled": False,
        "use_cache": USE_CACHE,
        "logits_to_keep": LOGITS_TO_KEEP,
    }


def prepare_decode_inputs(
    model: Qwen2ForCausalLM, inputs: BatchEncoding, device: str,
) -> tuple[torch.Tensor, torch.Tensor, Cache]:
    """Build a fresh prompt cache and matching one-token decode inputs."""
    prefill_outputs = model(**inputs, use_cache=USE_CACHE, logits_to_keep=LOGITS_TO_KEEP)
    cache = prefill_outputs.past_key_values
    if cache is None:
        raise RuntimeError("Expected prefill to return past_key_values.")
    next_tokens = prefill_outputs.logits.argmax(dim=-1)
    decode_attention_mask = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones(
                (inputs["input_ids"].shape[0], DECODE_TOKENS_PER_REQUEST),
                dtype=inputs["attention_mask"].dtype,
                device=device,
            ),
        ],
        dim=1,
    )
    return next_tokens, decode_attention_mask, cache
