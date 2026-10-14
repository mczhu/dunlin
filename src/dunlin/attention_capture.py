"""Versioned configuration for the bounded Nsight attention experiment.

This is data shared by collection and analysis, not a source archive. Git source
identity and cleanliness records remain separate capture-provenance artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


CAPTURE_CONFIG = "capture-config.json"
KERNEL_ORDER = ("qk", "kv_expand", "scale", "av", "kv_convert")
COUNTER_METRICS = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)


def validate_capture_config(config: Any) -> dict[str, Any]:
    """Validate supported v1 collection semantics, including subset/preflight runs."""
    try:
        if (not isinstance(config, dict) or type(config.get("schema_version")) is not int
                or config["schema_version"] != 1
                or config.get("kind") != "attention_bandwidth_capture"):
            raise ValueError("Unsupported capture configuration schema")
        if (config["source_mode"] not in ("strict", "debug")
                or config["record_origin"] not in ("capture_startup", "legacy_migration")
                or type(config["preflight"]) is not bool):
            raise ValueError("Invalid capture mode/origin")
        if config["record_origin"] == "legacy_migration":
            migration = config["migration"]
            if (config["source_mode"] != "strict"
                    or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", migration["source_git_commit"])
                    or not re.fullmatch(r"[0-9a-f]{64}", migration["source_script_sha256"])
                    or migration["source_script_path"] != "experiments/profile_attention_bandwidth.py"
                    or not isinstance(migration["input_sha256"], dict) or not migration["input_sha256"]
                    or any(not re.fullmatch(r"[0-9a-f]{64}", digest)
                           for digest in migration["input_sha256"].values())):
                raise ValueError("Invalid migration provenance")
        capture = config["capture"]
        expected = {"device": "0", "nvtx_include": "cached_decode_forward/", "range_filter": "::2",
                    "kernel_name_base": "demangled", "rename_kernels": "off", "launch_count": 1,
                    "replay_mode": "application", "cache_control": "none", "clock_control": "none"}
        if any(capture.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported capture selection/replay settings")
        if type(capture["launch_count"]) is not int:
            raise ValueError("Expected one selected launch")
        sections = capture["sections"]
        supported_sections = ["LaunchStats", "SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy"]
        if sections not in (["LaunchStats"], supported_sections):
            raise ValueError("Unsupported capture sections")
        if config["preflight"]:
            if sections != ["LaunchStats"] or capture["metrics"] != []:
                raise ValueError("Preflight must collect LaunchStats only")
        elif capture["metrics"] not in (list(COUNTER_METRICS), []):
            raise ValueError("Unsupported counter metrics")
        elif not capture["metrics"] and sections != supported_sections:
            raise ValueError("Counter capture must request metrics or detailed sections")
        workload = config["workload"]
        if (workload["runner"] != "experiments/profile_forward_phases.py"
                or workload["phase"] != "cached_decode"
                or workload["torch_profiler_enabled"] is not False
                or not isinstance(workload["python"], str) or not workload["python"]
                or not isinstance(workload["baseline_runs"], dict)):
            raise ValueError("Unsupported capture workload")
        points = config["points"]
        if not isinstance(points, list) or not points:
            raise ValueError("No configured capture points")
        identities = set()
        batches = set()
        for point in points:
            kernel, batch = point["kernel"], point["batch_size"]
            if (kernel not in KERNEL_ORDER or type(batch) is not int or batch not in (16, 32)
                    or (kernel, batch) in identities):
                raise ValueError("Invalid or duplicate capture point")
            identities.add((kernel, batch))
            batches.add(str(batch))
            pattern, skip, grid = point["selector_regex"], point["launch_skip"], point["expected_grid"]
            if (not isinstance(pattern, str) or not pattern or type(skip) is not int or skip < 0
                    or not isinstance(grid, list) or len(grid) != 3
                    or any(type(n) is not int or n <= 0 for n in grid)):
                raise ValueError("Invalid captured selector")
            re.compile(pattern)
        baselines = workload["baseline_runs"]
        if (set(baselines) != batches
                or any(not isinstance(path, str) or not path for path in baselines.values())):
            raise ValueError("Missing or unexpected capture baseline references")
    except (KeyError, TypeError, AttributeError, re.error) as error:
        raise ValueError("Malformed capture configuration") from error
    return config


def read_capture_config(session: Path) -> dict[str, Any]:
    path = session / CAPTURE_CONFIG
    if not path.is_file():
        raise ValueError("Missing capture-config.json; legacy sessions must first use "
                         "experiments/migrate_attention_capture_config.py")
    return validate_capture_config(json.loads(path.read_text()))


def write_capture_config(session: Path, config: dict[str, Any]) -> None:
    validate_capture_config(config)
    with (session / CAPTURE_CONFIG).open("x") as handle:
        handle.write(json.dumps(config, indent=2) + "\n")


def capture_point(config: dict[str, Any], kernel: str, batch: int) -> dict[str, Any]:
    for point in config["points"]:
        if point["kernel"] == kernel and point["batch_size"] == batch:
            return point
    raise ValueError(f"Unconfigured capture point: {kernel} B={batch}")
