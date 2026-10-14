"""Resolve benchmark devices and synchronize their queued work."""

from __future__ import annotations

import torch


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def select_device(requested_device: str = "auto", requested_dtype: str = "auto") -> tuple[str, torch.dtype]:
    """Resolve explicit choices, preserving the original MPS/CUDA/CPU auto policy."""
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    elif requested_device == "cpu":
        device = "cpu"
    elif requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        device = "mps"
    elif requested_device == "cuda" or requested_device.startswith("cuda:"):
        index = None
        if requested_device != "cuda":
            index_text = requested_device.removeprefix("cuda:")
            if not index_text.isdecimal():
                raise ValueError(f"Invalid CUDA device index: {requested_device}")
            index = int(index_text)
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if index is None:
            device = "cuda"
        else:
            if index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {index} is unavailable; found {torch.cuda.device_count()} device(s)")
            device = f"cuda:{index}"
    else:
        raise ValueError(f"Unknown device: {requested_device}")

    if requested_dtype == "auto":
        dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        try:
            dtype = DTYPES[requested_dtype]
        except KeyError as error:
            raise ValueError(f"Unknown dtype: {requested_dtype}") from error
    return device, dtype


def synchronize(device: str) -> None:
    """Wait for work on the selected accelerator before reading a timestamp."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elif device.startswith("cuda:"):
        torch.cuda.synchronize(device)
