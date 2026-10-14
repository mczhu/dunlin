"""Attribute saved Chrome trace device events to SDPA; no inference required."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
CONTAINMENT_TOLERANCE_US = 0.01  # Export rounding, not timing uncertainty.


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    """Count overlap once; input timestamps and result are in microseconds."""
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise ValueError("Expected finite, ordered interval endpoints")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _interval(event: dict[str, Any]) -> tuple[float, float]:
    start, duration = float(event["ts"]), float(event["dur"])
    if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
        raise ValueError("Invalid complete-event timestamp or duration")
    return start, start + duration


def _inside(event: dict[str, Any], parent: dict[str, Any]) -> bool:
    start, end = _interval(event)
    left, right = _interval(parent)
    return (event["pid"] == parent["pid"] and event["tid"] == parent["tid"]
            and left - CONTAINMENT_TOLERANCE_US <= start
            and end <= right + CONTAINMENT_TOLERANCE_US)


def decode_flops_per_layer_per_request(config: dict[str, Any]) -> dict[str, int]:
    """Leading matrix FLOPs for one cached-decode token; multiply-add = two."""
    if config["phase"] != "cached_decode" or config["decode_tokens_per_request"] != 1:
        raise ValueError("FLOP model requires one-token cached decode")
    architecture = config["architecture"]
    hidden = architecture["hidden_size"]
    intermediate = architecture["intermediate_size"]
    query_heads = architecture["num_attention_heads"]
    kv_heads = architecture["num_key_value_heads"]
    head_dim = architecture["head_dim"]
    available_kv = config["input_tokens_per_request"] + 1
    return {
        "attention_products": 4 * query_heads * head_dim * available_kv,
        "qkvo_projections": 4 * hidden * head_dim * (query_heads + kv_heads),
        "mlp_projections": 6 * hidden * intermediate,
    }


def rank_device_kernels(
    selected: list[dict[str, Any]], operators: dict[int, dict[str, Any]],
    attention_owner: dict[int, int], cpu_ops: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank exact kernel variants and retain inspectable host ancestry.

    A variant includes scope, full name, launch geometry, and CPU ancestry.
    Duration sums rank execution cost; they are not elapsed time with overlap.
    No tensor identity, dtype, or mathematical role is inferred here.
    """
    kernels = sorted((e for e in selected if e.get("cat") == "kernel"),
                     key=lambda e: (float(e["ts"]), e["args"]["External id"]))
    paths = {}
    for external_id in {e["args"]["External id"] for e in kernels}:
        op = operators[external_id]
        parents = sorted((p for p in cpu_ops if _inside(op, p)),
                         key=lambda p: (float(p["ts"]), -float(p["dur"]),
                                        p["args"]["External id"]))
        paths[external_id] = [{"name": p["name"], "external_id": p["args"]["External id"],
                               "ts_us": float(p["ts"]), "duration_us": float(p["dur"])}
                              for p in parents]
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    launches = []
    for ordinal, event in enumerate(kernels):
        args = event["args"]
        external_id = args["External id"]
        call = attention_owner.get(external_id)
        launch = {
            "forward_kernel_ordinal": ordinal, "kernel_name": event["name"],
            "grid": args.get("grid"), "block": args.get("block"),
            "ts_us": float(event["ts"]), "duration_us": float(event["dur"]),
            "external_id": external_id, "correlation": args.get("correlation"),
            "sdpa_call_index": call, "direct_owner": operators[external_id]["name"],
            "cpu_ancestry": paths[external_id],
        }
        launches.append(launch)
        key = ("sdpa" if call is not None else "outside_sdpa", event["name"],
               tuple(args.get("grid", [])), tuple(args.get("block", [])),
               tuple(p["name"] for p in paths[external_id]))
        grouped[key].append(launch)
    totals = {
        "forward": sum(e["duration_us"] for e in launches),
        "sdpa": sum(e["duration_us"] for e in launches if e["sdpa_call_index"] is not None),
    }
    rows = []
    for (scope, name, grid, block, path), xs in grouped.items():
        durations = [e["duration_us"] for e in xs]
        total = sum(durations)
        rows.append({
            "scope": scope, "kernel_name": name, "grid": list(grid), "block": list(block),
            "direct_owner": xs[0]["direct_owner"], "cpu_ancestry_names": list(path),
            "launch_count": len(xs), "duration_sum_ms": total / 1000,
            "duration_union_ms": interval_union_us([
                (e["ts_us"], e["ts_us"] + e["duration_us"]) for e in xs]) / 1000,
            "mean_us": statistics.mean(durations), "median_us": statistics.median(durations),
            "min_us": min(durations), "max_us": max(durations),
            "share_of_forward_kernel_duration_sum": (total / totals["forward"]
                                                      if totals["forward"] else None),
            "share_of_sdpa_kernel_duration_sum": (total / totals["sdpa"]
                                                   if scope == "sdpa" and totals["sdpa"] else None),
            "sdpa_call_count": len({e["sdpa_call_index"] for e in xs
                                    if e["sdpa_call_index"] is not None}),
            "first_launch": xs[0],
        })
    rows.sort(key=lambda row: (-row["duration_sum_ms"], row["scope"], row["kernel_name"],
                               row["grid"], row["block"], row["cpu_ancestry_names"]))
    return {
        "forward_kernel_duration_sum_ms": totals["forward"] / 1000,
        "sdpa_kernel_duration_sum_ms": totals["sdpa"] / 1000,
        "rows": rows,
        "first_sdpa_launches": [e for e in launches if e["sdpa_call_index"] == 0],
    }


def analyze_attention_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze exactly one host forward and its following completion wait.

    Attribute device events through External id, never GPU/CPU time overlap.
    Unknown device activity inside the completion window is rejected rather
    than silently counted as forward work. Other work outside it is ignored.
    """
    complete = [event for event in events if event.get("ph") == "X"]

    def marker(name: str) -> dict[str, Any]:
        found = [e for e in complete if e.get("cat") == "user_annotation" and e["name"] == name]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {name} marker")
        return found[0]

    forward, wait = marker("cached_decode_forward"), marker("wait_for_completion")
    f0, f1 = _interval(forward)
    w0, w1 = _interval(wait)
    if (forward["pid"], forward["tid"]) != (wait["pid"], wait["tid"]) or not f0 <= f1 <= w0 < w1:
        raise ValueError("Forward and wait must be ordered on the same host thread")
    cpu_ops = [e for e in complete if e.get("cat") == "cpu_op" and _inside(e, forward)]
    operators = {}
    for op in cpu_ops:
        external_id = op.get("args", {}).get("External id")
        if external_id is None or external_id in operators:
            raise ValueError("Selected CPU operators require unique External ids")
        operators[external_id] = op
    attention = sorted((e for e in cpu_ops if e["name"] == "aten::scaled_dot_product_attention"),
                       key=lambda e: float(e["ts"]))
    if not attention:
        raise ValueError("No scaled_dot_product_attention operators in forward")
    attention_owner = {}
    for external_id, op in operators.items():
        matches = [i for i, parent in enumerate(attention) if _inside(op, parent)]
        if len(matches) > 1:
            raise ValueError("Ambiguous nested/overlapping SDPA ranges")
        if matches:
            attention_owner[external_id] = matches[0]

    selected, ignored = [], 0
    for event in complete:
        if event.get("cat") not in DEVICE_CATEGORIES:
            continue
        start, end = _interval(event)
        external_id = event.get("args", {}).get("External id")
        if external_id in operators:
            if start < f0 or end > w1:
                raise ValueError("Correlated forward device event extends beyond completion window")
            selected.append(event)
        elif start < w1 and end > f0:
            raise ValueError("Unattributed device activity overlaps completion window")
        else:
            ignored += 1
    if not selected:
        raise ValueError("No correlated device activity")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    owner_durations: dict[str, float] = defaultdict(float)
    attention_by_call: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for event in selected:
        external_id = event["args"]["External id"]
        if external_id in attention_owner:
            name = operators[external_id]["name"]
            group = name if name in {"aten::copy_", "aten::mul", "aten::bmm"} else "attention_other"
            owner_durations[name] += float(event["dur"])
            attention_by_call[attention_owner[external_id]].append(_interval(event))
        else:
            group = "outside_sdpa"
        groups[group].append(event)
    all_intervals = [_interval(e) for e in selected]
    attention_intervals = [interval for xs in attention_by_call.values() for interval in xs]
    activity = interval_union_us(all_intervals)
    attention_activity = interval_union_us(attention_intervals)
    per_call_gaps = []
    for i in range(len(attention)):
        intervals = attention_by_call[i]
        if not intervals:
            raise ValueError("SDPA call has no correlated device events")
        span = max(e for _, e in intervals) - min(s for s, _ in intervals)
        per_call_gaps.append(span - interval_union_us(intervals))
    group_union = {name: interval_union_us([_interval(e) for e in xs]) / 1000
                   for name, xs in sorted(groups.items())}
    streams = Counter((e["args"].get("device"), e["args"].get("context"), e["args"].get("stream"))
                      for e in selected)
    return {
        "kernel_ranking": rank_device_kernels(selected, operators, attention_owner, cpu_ops),
        "completion_ms": (w1 - f0) / 1000,
        "host_forward_ms": (f1 - f0) / 1000,
        "wait_ms": (w1 - w0) / 1000,
        "device_activity_ms": activity / 1000,
        "forward_activity_gap_ms": (w1 - f0 - activity) / 1000,
        "attention_activity_ms": attention_activity / 1000,
        "attention_share_of_device_activity": attention_activity / activity,
        "outside_sdpa_activity_ms": interval_union_us([
            _interval(e) for e in groups["outside_sdpa"]]) / 1000,
        "group_activity_ms": group_union,
        "group_overlap_excess_ms": max(0.0, sum(group_union.values()) - activity / 1000),
        "device_event_overlap_excess_ms": max(0.0, sum(e["dur"] for e in selected) - activity) / 1000,
        "attention_direct_owner_duration_sums_ms": {name: value / 1000
                                                    for name, value in sorted(owner_durations.items())},
        "attention_internal_gap_sum_ms": sum(per_call_gaps) / 1000,
        "sdpa_call_count": len(attention),
        "attention_device_event_count": sum(len(xs) for xs in attention_by_call.values()),
        "selected_device_event_count": len(selected),
        "ignored_device_event_count": ignored,
        "active_streams": [{"device": d, "context": c, "stream": s, "event_count": count}
                           for (d, c, s), count in streams.items()],
        "cpu_operator_counts": dict(sorted(Counter(e["name"] for e in cpu_ops).items())),
        "attention_kernel_name_counts": dict(sorted(Counter(
            e["name"] for e in selected if e.get("cat") == "kernel"
            and e["args"]["External id"] in attention_owner).items())),
    }


def analyze_attention_run(run_dir: Path, *, require_strict: bool = False) -> dict[str, Any]:
    """Read raw evidence and retain capture metadata and content hashes."""
    names = ["trace.json", "config.resolved.json", "environment.json", "manifest.json", "validation.json"]
    if (run_dir / "source-checks.jsonl").is_file():
        names.append("source-checks.jsonl")
    raw = {name: (run_dir / name).read_bytes() for name in names}
    records = {name: json.loads(value) for name, value in raw.items() if name.endswith(".json")}
    if any(not isinstance(record, dict) for record in records.values()):
        raise ValueError("Expected object-valued trace records")
    manifest, config = records["manifest.json"], records["config.resolved.json"]
    if "source-checks.jsonl" in manifest.get("artifacts", []) and "source-checks.jsonl" not in raw:
        raise ValueError("Missing recorded trace source checks")
    if manifest.get("status") != "complete" or config.get("phase") != "cached_decode":
        raise ValueError("Expected a complete cached-decode capture")
    environment = records["environment.json"]
    source_checks = [json.loads(line) for line in raw.get("source-checks.jsonl", b"").splitlines()
                     if line.strip()]
    if "source-checks.jsonl" in raw and not source_checks:
        raise ValueError("Empty recorded source checks")
    source_records = [manifest, config, environment, *source_checks]
    if any(not isinstance(record, dict) for record in source_records):
        raise ValueError("Expected object-valued trace source records")
    if any(record["source_mode"] not in ("strict", "debug")
           for record in source_records if "source_mode" in record):
        raise ValueError("Unknown recorded trace source mode")
    modes = {record["source_mode"] for record in source_records if "source_mode" in record}
    if len(modes) > 1:
        raise ValueError("Recorded trace source modes disagree")
    # Historical captures predate source_mode. Never infer strict from cleanliness
    # when any record explicitly labels the input as development-only.
    debug = "debug" in modes or any(record.get("publication_eligible") is False for record in source_records)
    if require_strict and debug:
        raise ValueError("Official trace analysis rejects debug inputs")
    if not debug:
        if environment.get("git_dirty") is not False:
            raise ValueError("Strict trace inputs require a recorded clean capture checkout")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(environment.get("git_commit", ""))):
            raise ValueError("Strict trace inputs require a recorded full capture commit")
        for check in source_checks:
            if (check.get("passed") is not True or check.get("accepted") is not True
                    or check.get("git_status") != ""
                    or check.get("git_commit") != environment.get("git_commit")
                    or check.get("expected_commit") != environment.get("git_commit")):
                raise ValueError("Recorded strict trace source check failed or changed commit")
    validation = records["validation.json"]
    if not all(validation.get(key) is True for key in
               ("profiled_matches_unprofiled", "forward_matches_reference", "logits_finite")):
        raise ValueError("Capture did not pass profiled/unprofiled output correctness validation")
    trace = records["trace.json"]
    return {
        "run_id": manifest["run_id"], "source_run_dir": str(run_dir),
        "batch_size": config["batch_size"],
        "source_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        "config": config, "environment": environment,
        "manifest": manifest, "validation": validation,
        "source_mode": "debug" if debug else "strict", "publication_eligible": not debug,
        "source_checks": source_checks,
        "flops_per_layer_per_request": decode_flops_per_layer_per_request(config),
        "metrics": analyze_attention_events(trace["traceEvents"]),
    }
