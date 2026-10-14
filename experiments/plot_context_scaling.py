"""Plot context-length scaling at one batch size from a completed saved matrix."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dunlin.artifacts import write_json
from dunlin.matrix_results import load_matrix_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = load_matrix_run(args.run_dir)
    if args.batch_size not in run.config["batch_sizes"]:
        parser.error("batch size must be present in the saved matrix")
    output = args.output or args.run_dir / f"context_scaling_b{args.batch_size}.png"
    if output.exists() and args.output is None:
        parser.error(f"Plot already exists: {output}; select an explicit --output")

    contexts = sorted(run.config["context_lengths"])
    entries = {
        point["context_length"]: point
        for point in run.manifest["matrix_points"]
        if point["batch_size"] == args.batch_size
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    table: list[dict[str, float | int | str]] = []
    for column, (phase, label, color) in enumerate((
        ("prefill", "Prefill", "tab:blue"),
        ("cached_decode", "One-token cached decode", "tab:orange"),
    )):
        medians, lower, upper, throughput = [], [], [], []
        for context in contexts:
            child = args.run_dir / entries[context]["run_id"]
            with (child / "measurements.csv").open(newline="") as source:
                samples = [
                    float(row["latency_seconds"])
                    for row in csv.DictReader(source) if row["phase"] == phase
                ]
            median = statistics.median(samples)
            q25, _, q75 = statistics.quantiles(samples, n=4, method="inclusive")
            tokens = args.batch_size * (context if phase == "prefill" else 1)
            medians.append(median * 1000)
            lower.append(q25 * 1000)
            upper.append(q75 * 1000)
            throughput.append(tokens / median)
            table.append({
                "context_length": context, "batch_size": args.batch_size,
                "phase": phase, "samples": len(samples),
                "median_latency_ms": median * 1000,
                "q25_latency_ms": q25 * 1000, "q75_latency_ms": q75 * 1000,
                "tokens_per_second": tokens / median,
            })
        latency_axis, throughput_axis = axes[:, column]
        latency_axis.plot(contexts, medians, "o-", color=color)
        latency_axis.fill_between(contexts, lower, upper, alpha=0.2, color=color)
        latency_axis.set_title(f"{label}: latency")
        latency_axis.set_ylabel("Median forward latency (ms)")
        throughput_axis.plot(contexts, throughput, "o-", color=color)
        throughput_axis.set_title(f"{label}: throughput")
        throughput_axis.set_ylabel(
            "Input tokens/s" if phase == "prefill" else "Output tokens/s"
        )
        for context, median in zip(contexts, medians, strict=True):
            latency_axis.annotate(f"{median:.1f}", (context, median),
                                  xytext=(0, 7), textcoords="offset points", ha="center")
    for axis in axes.flat:
        axis.set_xticks(contexts)
        axis.set_ylim(0, axis.dataLim.ymax * 1.2)
        axis.grid(True, alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("Prompt context length N (tokens per request)")
    figure.suptitle(
        f"{run.config['model_id']} | {run.environment['device_name']} | "
        f"{run.config['resolved_dtype']} | B={args.batch_size}"
    )
    figure.text(0.5, 0.025,
                f"Source: {run.run_id}\n"
                "Shading: sample interquartile range, not a confidence interval. "
                "Throughput = batch tokens / median latency.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    write_json(output.with_suffix(".sources.json"), {
        "kind": "saved_benchmark_samples",
        "source_run_ids": [run.run_id],
        "source_child_run_ids": [entries[context]["run_id"] for context in contexts],
        "batch_size": args.batch_size,
        "latency_statistic": "median",
        "spread": "inclusive sample quartiles; not confidence intervals",
        "throughput_definitions": {
            "prefill": "batch_size * context_length / median_prefill_latency_seconds",
            "cached_decode": "batch_size / median_cached_decode_latency_seconds",
        },
        "points": table,
    })
    print(f"plot_path: {output}")
    print(f"sources_path: {output.with_suffix('.sources.json')}")


if __name__ == "__main__":
    main()
