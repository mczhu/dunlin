"""Plot saved decode latency and throughput as zero-based bars with sample spread."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dunlin.artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Output of analyze_decode_scaling.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("kind") != "fixed_context_decode_scaling_analysis":
        parser.error("Expected a saved decode scaling analysis")
    if args.output.exists() or args.output.with_suffix(".sources.json").exists():
        parser.error("Choose a new output path")
    points = sorted(summary["points"], key=lambda point: point["batch_size"])
    batches = [point["batch_size"] for point in points]
    positions = list(range(len(points)))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    medians = [point["median_seconds"] * 1000 for point in points]
    latency_bars = axes[0].bar(positions, medians, width=0.55, color="tab:blue")
    axes[0].errorbar(positions, medians, yerr=[
        [median - point["p10_seconds"] * 1000 for median, point in zip(medians, points)],
        [point["p90_seconds"] * 1000 - median for median, point in zip(medians, points)],
    ], fmt="none", ecolor="black", capsize=4, label="Inclusive sample p10–p90")
    axes[0].bar_label(latency_bars, fmt="%.2f", padding=6)
    axes[0].set_ylim(0, max(point["p90_seconds"] * 1000 for point in points) * 1.25)
    axes[0].set_ylabel("Synchronized decode batch latency (ms)")
    axes[0].legend(fontsize="small")
    throughput = [point["tokens_per_second"] for point in points]
    throughput_bars = axes[1].bar(positions, throughput, width=0.55, color="tab:blue")
    axes[1].bar_label(throughput_bars, fmt="%.2f", padding=4)
    axes[1].set_ylim(0, max(throughput) * 1.25)
    axes[1].set_ylabel("Aggregate decoded tokens/s (B / median latency)")
    axes[0].set_title("Median batch latency")
    axes[1].set_title("Decode throughput")
    if len(points) == 2:
        comparison = f"B={batches[0]}→{batches[1]}"
        axes[0].set_title(f"Median batch latency\n{medians[1] / medians[0] - 1:+.1%} for {comparison}")
        axes[1].set_title(f"Decode throughput\n{throughput[1] / throughput[0] - 1:+.1%} for {comparison}")
    for axis in axes:
        axis.set_xlabel("Batch size")
        axis.set_xticks(positions, batches)
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)
    debug = summary.get("source_mode") == "debug"
    title = f"{summary['environment']['device_name']}, N={summary['config']['input_tokens_per_request']}"
    figure.suptitle(("DEBUG — development results: " if debug else "") + title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    provenance = {
        "analysis_path": str(args.summary),
        "analysis_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "source_run_ids": summary["source_run_ids_in_supplied_order"],
        "spread": summary["metric_definitions"]["spread"],
        "scope": "Median synchronized decode latency and aggregate throughput; latency whiskers are sample p10–p90, not confidence intervals",
        "source_mode": "debug" if debug else "strict",
        "publication_eligible": not debug,
    }
    if debug:
        provenance.update(source_mode="debug", publication_eligible=False)
    write_json(args.output.with_suffix(".sources.json"), provenance)
    print(f"plot_path: {args.output}")


if __name__ == "__main__":
    main()
