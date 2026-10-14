"""Plot saved attention counters; collection and analysis stay separate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dunlin.artifacts import write_json
from dunlin.attention_bandwidth import KERNEL_ORDER, sha256


def validate_dram_source(summary: dict, *, require_strict: bool) -> bool:
    """Validate selection, sampling, correctness and source for the standalone view."""
    captures = summary["captures"]
    expected = {(kernel, batch) for kernel in KERNEL_ORDER for batch in (16, 32)}
    if len(captures) != len(expected) or {(c["kernel"], c["batch_size"]) for c in captures} != expected:
        raise ValueError("Expected one counter selection per kernel at B=16 and B=32")
    records = [summary, *captures, *summary["configs_by_batch"].values()]
    for capture in captures:
        if capture["sample_count"] != 1 or capture["replay_passes"] < 1:
            raise ValueError("Expected one selected launch with recorded replay passes")
        value = capture["dram_pct"]
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("Expected finite DRAM percentage on the 0–100% axis")
        if len(capture["replay_checks"]) != capture["replay_passes"]:
            raise ValueError("Missing replay correctness records")
        for replay in capture["replay_checks"]:
            records.append(replay["manifest"])
            if replay["manifest"]["status"] != "complete" or not all(
                replay["validation"].get(key) is True for key in ("forward_matches_reference", "logits_finite")
            ):
                raise ValueError("Expected complete, correctness-validated counter replays")
    debug = any(record.get("source_mode", "strict") == "debug"
                or record.get("publication_eligible") is False for record in records)
    if require_strict and debug:
        raise ValueError("Official DRAM plot rejects debug inputs")
    if not debug and summary["environment"].get("git_dirty") is not False:
        raise ValueError("Strict inputs require a recorded clean capture checkout")
    return debug


def plot_dram_throughput(axis: plt.Axes, summary: dict) -> None:
    """Plot exported percentages without captions or explanatory footnotes."""
    roles = [("qk", "QKᵀ product"), ("av", "AV product"),
             ("kv_expand", "First KV-head expansion"),
             ("scale", "Expanded-operand scaling"),
             ("kv_convert", "First input conversion")]
    rows = {(row["kernel"], row["batch_size"]): row for row in summary["captures"]}
    for batch, offset, color in ((16, -0.18, "#6A51A3"), (32, 0.18, "#B39CD0")):
        bars = axis.barh([i + offset for i in range(len(roles))],
                         [rows[kernel, batch]["dram_pct"] for kernel, _ in roles],
                         height=0.32, color=color, label=f"B={batch}")
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=13)
    axis.set_yticks(range(len(roles)), [label for _, label in roles])
    axis.invert_yaxis()
    axis.set_ylim(len(roles) + 0.35, -0.5)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Nsight sustained DRAM throughput reference (%)")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", fontsize=12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--panel", choices=("all", "dram-throughput"), default="all",
                        help="Render the original four panels or a reusable standalone DRAM plot")
    parser.add_argument("--require-strict", action="store_true",
                        help="Reject debug inputs in the standalone DRAM plot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("kind") != "attention_bandwidth_analysis":
        parser.error("Expected an attention bandwidth analysis")
    sources = args.output.with_suffix(".sources.json")
    if args.output.exists() or sources.exists():
        parser.error("Choose a new output path")
    standalone = args.panel == "dram-throughput"
    if standalone:
        try:
            debug = validate_dram_source(summary, require_strict=args.require_strict)
        except (ValueError, KeyError) as error:
            parser.error(str(error))
        plt.rcParams.update({"font.size": 14, "axes.titlesize": 16})
        figure, axis = plt.subplots(figsize=(9, 5.5))
        plot_dram_throughput(axis, summary)
        contexts = {config["input_tokens_per_request"] for config in summary["configs_by_batch"].values()}
        if len(contexts) != 1:
            parser.error("Chart requires matching context lengths")
        axis.set_title(f"{'DEBUG — ' if debug else ''}{summary['environment']['device_name']} cached decode, "
                       f"N={next(iter(contexts))}\nSelected-kernel DRAM throughput")
        figure.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)
        write_json(sources, {
            "summary_path": str(args.summary), "summary_sha256": sha256(args.summary),
            "source_session_id": summary["source_session_id"],
            "plot_script_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
            "source_mode": "debug" if debug else "strict", "publication_eligible": not debug,
            "scope": "One selected launch per inferred role/batch from a separate counter capture; exported DRAM throughput percent; replay passes are not independent samples",
        })
        print(f"plot_path: {args.output}")
        return
    if args.require_strict:
        parser.error("--require-strict requires --panel dram-throughput")
    captures = summary["captures"]
    x = list(range(len(KERNEL_ORDER)))
    colors = {16: "#326a96", 32: "#d17736"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("duration_us", "Selected kernel duration (µs)"),
              ("dram_total_bytes", "DRAM read + write traffic (decimal MB)"),
              ("dram_gb_per_second", "Achieved DRAM bandwidth (decimal GB/s)"),
              ("dram_pct", "DRAM throughput (% of sustained peak)")]
    for axis, (field, title) in zip(axes.flat, panels):
        for batch, offset in ((16, -0.19), (32, 0.19)):
            rows = [next(c for c in captures if c["kernel"] == k and c["batch_size"] == batch)
                    for k in KERNEL_ORDER]
            values = [row[field] / (1e6 if field == "dram_total_bytes" else 1) for row in rows]
            bars = axis.bar([value + offset for value in x], values, width=0.36,
                            color=colors[batch], label=f"B={batch}")
            axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(x, ["QKᵀ", "KV expansion", "Scale", "AV", "KV conversion"])
        axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0, 0].legend()
    debug = summary.get("source_mode") == "debug"
    title = ("DEBUG — development attention counters, N=500" if debug else
             "T4 math attention, N=500: doubled traffic and duration; similar bandwidth")
    figure.suptitle(title, fontsize=14)
    passes = ", ".join(str(value) for value in sorted({row["replay_passes"] for row in captures}))
    figure.text(0.5, 0.015, f"One selected launch per kernel/batch, collected across {passes} application replay passes. "
                "No independent repeats or error bars.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    provenance = {
        "source_session_id": summary["source_session_id"], "summary_sha256": sha256(args.summary),
        "plot_script_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "scope": summary["metric_definitions"]["sample_count"],
    }
    if debug:
        provenance.update(source_mode="debug", publication_eligible=False)
    write_json(sources, provenance)
    print(f"plot_path: {args.output}")


if __name__ == "__main__":
    main()
