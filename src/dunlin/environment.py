"""Collect provenance without affecting measured forward intervals."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            check=True, timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _system_memory_bytes() -> int | None:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True,
                text=True, check=True, timeout=5,
            )
            return int(result.stdout.strip())
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_environment(device: str, dtype: torch.dtype) -> dict[str, object]:
    """Return available source, software, hardware, and runtime settings for one run.

    Captures:
    - Source provenance (Git commit, dirty status, uv.lock SHA).
    - Software versions (Python, PyTorch, Transformers, CUDA build).
    - Host hardware details (OS, CPU architecture, physical memory, GPU specs).
    - Runtime performance and precision controls:
      * Environment variables (CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS, MPS fallback).
      * PyTorch CPU thread counts (intra-op and inter-op threads).
      * Matrix multiplication precision controls (float32_matmul_precision, TF32 flags).
    """
    lockfile = ROOT / "uv.lock"
    try:
        lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    except OSError:
        lock_hash = None
    worktree_status = _git("status", "--porcelain", "--untracked-files=all")
    system_memory = _system_memory_bytes()
    unavailable_fields: dict[str, str] = {}
    if system_memory is None:
        unavailable_fields["system_physical_memory_bytes"] = "OS memory query unavailable"
    record: dict[str, object] = {
        "schema_version": 2,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if worktree_status is None else bool(worktree_status),
        "uv_lock_sha256": lock_hash,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_processor": platform.processor() or None,
        "system_physical_memory_bytes": system_memory,
        "wsl": "microsoft" in platform.release().lower() if sys.platform == "linux" else False,
        "backend": device.split(":", 1)[0],
        "device": device,
        "dtype": str(dtype),
        "torch_cuda_build_version": torch.version.cuda,
        "runtime_settings": {
            **{
                name: os.environ.get(name)
                for name in ("CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK", "OMP_NUM_THREADS")
            },
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
                if hasattr(torch, "get_float32_matmul_precision")
                else None
            ),
            "cuda_matmul_allow_tf32": getattr(
                getattr(getattr(torch.backends, "cuda", None), "matmul", None),
                "allow_tf32",
                None,
            ),
            "cudnn_allow_tf32": getattr(
                getattr(torch.backends, "cudnn", None),
                "allow_tf32",
                None,
            ),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": getattr(
                getattr(getattr(torch.backends, "cuda", None), "matmul", None),
                "allow_fp16_reduced_precision_reduction",
                None,
            ),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": getattr(
                getattr(getattr(torch.backends, "cuda", None), "matmul", None),
                "allow_bf16_reduced_precision_reduction",
                None,
            ),
        },
    }
    if device.startswith("cuda"):
        index = torch.cuda.current_device() if device == "cuda" else int(device.split(":", 1)[1])
        properties = torch.cuda.get_device_properties(index)
        record["device_name"] = properties.name
        record["dedicated_device_memory_bytes"] = properties.total_memory
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits", "-i", str(index)],
                capture_output=True, text=True, check=True, timeout=5,
            )
            driver_version = driver.stdout.strip()
            if driver_version:
                record["nvidia_driver_version"] = driver_version
            else:
                record["nvidia_driver_version"] = None
                unavailable_fields["nvidia_driver_version"] = "nvidia-smi returned no driver version"
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            record["nvidia_driver_version"] = None
            unavailable_fields["nvidia_driver_version"] = "nvidia-smi query unavailable"
    elif device == "mps":
        record["device_name"] = f"Apple MPS ({platform.machine()})"
    else:
        record["device_name"] = platform.processor() or "CPU"
    if unavailable_fields:
        record["unavailable_fields"] = unavailable_fields
    return record
