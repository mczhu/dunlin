"""Plot saved matrix samples and optional comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dunlin.artifacts import write_json
from dunlin.matrix_results import compare_matrix_runs, load_matrix_run

def render_plot(
    series: list[tuple[str, list[dict[str, Any]]]], output: Path, title: str,
    source_label: str,
) -> None:
    """Render medians and the existing batch-one throughput-speedup formulas."""
    fields = (
        "median_prefill_forward_latency_ms",
        "median_cached_decode_forward_latency_ms",
        "prefill_throughput_speedup_vs_batch_one",
        "decode_throughput_speedup_vs_batch_one",
    )
    titles = (
        "Prefill batch latency", "One-token cached-decode batch latency",
        "Prefill throughput speedup vs. B=1", "Cached-decode throughput speedup vs. B=1",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    all_batches = sorted({int(point["batch_size"]) for _, points in series for point in points})
    for series_index, (label, points) in enumerate(series):
        contexts = sorted({int(point["context_length"]) for point in points})
        for context in contexts:
            context_points = sorted(
                (point for point in points if point["context_length"] == context),
                key=lambda point: point["batch_size"],
            )
            batches = [point["batch_size"] for point in context_points]
            for axis, field in zip(axes.flat, fields, strict=True):
                axis.plot(
                    batches, [point[field] for point in context_points], marker="o",
                    linestyle="-" if series_index == 0 else "--", label=f"{label}, N={context}",
                )
    for axis, heading in zip(axes.flat, titles, strict=True):
        axis.set_title(heading)
        axis.set_xlabel("Batch size")
        axis.set_xticks(all_batches)
        axis.set_ylabel("Median latency (ms)" if "latency" in heading else "Aggregate throughput speedup")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    figure.suptitle(title)
    figure.text(0.5, 0.02, source_label, ha="center", fontsize=7)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved matrix without running inference")
    parser.add_argument("run_dir", type=Path, help="Completed matrix run directory")
    parser.add_argument("--compare", type=Path, help="Second completed matrix run")
    parser.add_argument("--allow-cross-machine", action="store_true")
    parser.add_argument("--output", type=Path, help="PNG destination; defaults to RUN_DIR/plot.png")
    args = parser.parse_args()
    if args.compare is None and args.allow_cross_machine:
        parser.error("--allow-cross-machine requires --compare")
    if args.compare is not None and args.output is None:
        parser.error("--compare requires --output")
    baseline = load_matrix_run(args.run_dir)
    runs = [baseline]
    comparison = None
    if args.compare:
        candidate = load_matrix_run(args.compare)
        comparison = compare_matrix_runs(
            baseline, candidate, allow_cross_machine=args.allow_cross_machine,
            allow_debug=True,
        )
        runs.append(candidate)
    output = args.output or baseline.path / "plot.png"
    if output.exists() and args.output is None:
        parser.error(f"Plot already exists: {output}; pass --output to choose an explicit destination")
    labels = [
        f"{'baseline' if index == 0 else 'candidate'} "
        f"{'DEBUG ' if run.manifest.get('source_mode') == 'debug' else ''}{run.environment.get('backend')}"
        for index, run in enumerate(runs)
    ] if len(runs) > 1 else [str(baseline.environment.get("backend"))]
    debug = any(run.manifest.get("source_mode") == "debug" for run in runs)
    render_plot(
        [(label, run.summary["matrix_points"]) for label, run in zip(labels, runs, strict=True)],
        output, ("DEBUG — development results: " if debug else "")
        + f"{baseline.config['model_id']}: context length × static batch size",
        "Sources: " + " | ".join(run.run_id for run in runs),
    )
    provenance = {
        "kind": "saved_benchmark_samples",
        "source_run_ids": [run.run_id for run in runs],
        "source_child_run_ids": {
            run.run_id: run.summary["source_run_ids"] for run in runs
        },
        "comparison": comparison,
    }
    if debug:
        provenance.update(source_mode="debug", publication_eligible=False)
    write_json(output.with_suffix(".sources.json"), provenance)
    print(f"plot_path: {output}")
    print(f"sources_path: {output.with_suffix('.sources.json')}")


if __name__ == "__main__":
    main()
