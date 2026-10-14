"""Reproduce SDPA execution attribution from saved traces, without a GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

from dunlin.artifacts import write_json
from dunlin.trace_attention import analyze_attention_run


def kernel_label(name: str) -> str:
    """Short display label only; artifacts retain the full demangled name."""
    for marker in ("gemv2T_kernel_val", "internal::gemvx::kernel", "softmax_warp_forward"):
        if marker in name:
            return marker
    if "direct_copy_kernel_cuda" in name:
        return ("unrolled copy (cast-capable)" if "unrolled_elementwise_kernel" in name
                else "elementwise copy (no cast)")
    for marker in ("MulFunctor", "float16_copy_kernel_cuda", "where_kernel_impl",
                   "isneginf_kernel_impl", "and_kernel_cuda", "FillFunctor"):
        if marker in name:
            return marker
    return name.removeprefix("void ").replace("(anonymous namespace)::", "").split("(", 1)[0][:90]


def write_kernel_rankings(runs: list[dict], output: Path) -> None:
    batch_counts = Counter(run["batch_size"] for run in runs)
    debug = any(run["source_mode"] == "debug" for run in runs)
    title = "Kernel execution ranking from saved T4 decode traces"
    lines = [f"# {'DEBUG — ' if debug else ''}{title}", "",
             "This is a new analysis of existing measurements; no new GPU capture was taken.", "",
             "Variants are grouped by full kernel name, grid, block, SDPA membership, and CPU ancestry. "
             "Rank by summed device duration across the forward, not host operator duration. "
             "Repeated calls contribute to the total; mean duration describes one launch. "
             "With concurrency, sums and shares can count overlapping execution. "
             "Shares below use kernel duration sums, excluding copies/memsets that are not kernels, "
             "gaps, and wall time. Full names and first-launch correlation/ancestry evidence are in "
             "the CSVs and summary.json. Tensor roles and dtypes are not inferred by the analyzer.", ""]
    if debug:
        lines += ["DEBUG — development analysis; not eligible for publication.", ""]
    for run_index, run in enumerate(runs, 1):
        batch = run["batch_size"]
        ranking = run["metrics"]["kernel_ranking"]
        rows = ranking["rows"]
        suffix = f"_capture{run_index}" if batch_counts[batch] > 1 else ""
        filename = f"kernel_ranking_b{batch}{suffix}.csv"
        fields = ["rank", "scope", "kernel_name", "grid", "block", "direct_owner",
                  "cpu_ancestry_names", "launch_count", "duration_sum_ms", "duration_union_ms",
                  "mean_us", "median_us", "min_us", "max_us",
                  "share_of_forward_kernel_duration_sum", "share_of_sdpa_kernel_duration_sum",
                  "sdpa_call_count", "first_launch"]
        if debug:
            fields += ["source_mode", "publication_eligible"]
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rank, row in enumerate(rows, 1):
                record = {"rank": rank, **{key: json.dumps(value) if isinstance(value, (list, dict))
                                            else value for key, value in row.items()}}
                if debug:
                    record.update(source_mode="debug", publication_eligible=False)
                writer.writerow(record)
        lines += [f"## B={batch}", "", f"Source: `{run['run_id']}`. "
                  f"[All variants as CSV]({filename}).", "",
                  f"Forward kernel duration sum: {ranking['forward_kernel_duration_sum_ms']:.3f} ms; "
                  f"SDPA kernel duration sum: {ranking['sdpa_kernel_duration_sum_ms']:.3f} ms. "
                  f"Measured device overlap excess: {run['metrics']['device_event_overlap_excess_ms']:.6f} ms.", ""]
        for heading, subset in (("Largest forward variants (top 20)", rows[:20]),
                                ("All attention variants", [r for r in rows if r["scope"] == "sdpa"])):
            lines += [f"### {heading}", "",
                      "| Kernel variant | Owner | Scope | Grid | Calls | Total ms | Mean µs | Forward share | SDPA share |",
                      "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            for row in subset:
                attention_share = row["share_of_sdpa_kernel_duration_sum"]
                forward_share = row["share_of_forward_kernel_duration_sum"]
                lines.append(f"| {kernel_label(row['kernel_name'])} | `{row['direct_owner']}` | "
                             f"{row['scope']} | `{row['grid']}` | {row['launch_count']} | "
                             f"{row['duration_sum_ms']:.3f} | {row['mean_us']:.3f} | "
                             f"{format(forward_share, '.1%') if forward_share is not None else '—'} | "
                             f"{format(attention_share, '.1%') if attention_share is not None else '—'} |")
            lines.append("")
        lines += ["### First SDPA call: individual launch order", "",
                  "These are device launches linked to the first host SDPA call, ordered by device start. "
                  "The expansion pair has the same name/grid; tensor identity requires additional evidence.", "",
                  "| Order | Kernel variant | Grid | Duration µs | External id | Correlation | CPU ancestry |",
                  "| ---: | --- | --- | ---: | ---: | ---: | --- |"]
        for index, launch in enumerate(ranking["first_sdpa_launches"], 1):
            path = " → ".join(p["name"] for p in launch["cpu_ancestry"])
            lines.append(f"| {index} | {kernel_label(launch['kernel_name'])} | `{launch['grid']}` | "
                         f"{launch['duration_us']:.3f} | {launch['external_id']} | "
                         f"{launch['correlation']} | {path} |")
        lines.append("")
    (output / "kernel_ranking.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True, help="New analysis directory")
    parser.add_argument("--require-strict", action="store_true",
                        help="Reject debug captures for official trace analysis")
    args = parser.parse_args()
    try:
        runs = [analyze_attention_run(path, require_strict=args.require_strict) for path in args.run_dirs]
        if len({run["run_id"] for run in runs}) != len(runs):
            raise ValueError("Repeated source run")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    source = Path(__file__).resolve().parents[1]
    debug = any(run["source_mode"] == "debug" for run in runs)
    files = ["experiments/analyze_trace_attention.py", "src/dunlin/trace_attention.py"]
    summary = {
        "schema_version": 2, "kind": "trace_attention_attribution",
        "analysis_method": {
            "timestamp_unit": "microseconds (Chrome JSON ts/dur); output milliseconds",
            "window": "cached_decode_forward start through following wait_for_completion end",
            "attribution": "GPU External id -> CPU operator nested on the forward host thread; SDPA membership from CPU containment",
            "containment_tolerance_us": 0.01,
            "activity": "union of kernel, gpu_memcpy and gpu_memset execution intervals; excludes annotations",
            "share_denominator": "selected forward device activity, not completion wall time or hardware utilization",
            "owner_durations": "device duration sums by directly owning CPU operator; may double-count concurrent execution",
            "kernel_ranking": "full name + grid + block + SDPA membership + CPU ancestry; rank by duration sum; retain first launch and first SDPA sequence",
            "kernel_share_denominator": "kernel duration sum in forward or SDPA; excludes non-kernel device events and gaps; may count overlap",
            "flops": "leading matrix arithmetic per layer per request; multiply-add counts as two; excludes elementwise work and vocabulary head",
        },
        "analysis_environment": {"python": sys.version, "platform": platform.platform()},
        "analysis_source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                   for name in files},
        "source_mode": "debug" if debug else "strict", "publication_eligible": not debug,
        "runs": runs,
    }
    write_json(args.output_dir / "summary.json", summary)
    if debug:
        print("DEBUG — development analysis; not eligible for publication")
    write_kernel_rankings(runs, args.output_dir)
    for run in runs:
        metrics = run["metrics"]
        print(f"B={run['batch_size']}: SDPA {metrics['attention_activity_ms']:.3f} ms / "
              f"device activity {metrics['device_activity_ms']:.3f} ms "
              f"({metrics['attention_share_of_device_activity']:.1%}); "
              f"W={metrics['completion_ms']:.3f} ms")
    print(f"summary_path: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
