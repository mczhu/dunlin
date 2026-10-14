"""Plot saved SDPA execution and modeled arithmetic as reusable figures or a comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dunlin.artifacts import write_json


def validate_execution_source(summary: dict, *, require_strict: bool) -> bool:
    """Validate recorded source and correctness for either standalone trace plot."""
    runs = summary["runs"]
    if not runs:
        raise ValueError("Expected at least one analyzed run")
    records = [summary]
    for run in runs:
        records.extend((run, run["manifest"], run["config"], run["environment"]))
        if run["manifest"]["status"] != "complete" or run["config"]["phase"] != "cached_decode":
            raise ValueError("Expected complete cached-decode traces")
        if not all(run["validation"].get(key) is True for key in
                   ("profiled_matches_unprofiled", "forward_matches_reference", "logits_finite")):
            raise ValueError("Expected correctness-validated traces")
    if any(record.get("source_mode", "strict") not in ("strict", "debug") for record in records):
        raise ValueError("Unknown recorded execution source mode")
    debug = any(record.get("source_mode", "strict") == "debug"
                or record.get("publication_eligible") is False for record in records)
    if require_strict and debug:
        raise ValueError("Official execution plot rejects debug inputs")
    if not debug and any(run["environment"].get("git_dirty") is not False for run in runs):
        raise ValueError("Strict inputs require recorded clean capture checkouts")
    return debug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--panel", choices=("execution-arithmetic", "execution", "arithmetic"), default="execution-arithmetic",
                        help="Render the original comparison or either reusable standalone plot")
    parser.add_argument("--require-strict", action="store_true",
                        help="Reject debug inputs in any execution/arithmetic plot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("kind") != "trace_attention_attribution":
        parser.error("Expected a saved trace attention analysis")
    if args.output.exists() or args.output.with_suffix(".sources.json").exists():
        parser.error("Choose a new output path")
    standalone = args.panel != "execution-arithmetic"
    try:
        debug = validate_execution_source(summary, require_strict=args.require_strict)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    runs = summary["runs"]
    if not runs:
        parser.error("Expected at least one analyzed run")
    if args.panel != "arithmetic" and any(run["metrics"]["group_overlap_excess_ms"] > 1e-6 for run in runs):
        parser.error("Overlapping groups cannot form this additive stacked chart")
    if args.panel != "execution" and any(run["flops_per_layer_per_request"] != runs[0]["flops_per_layer_per_request"] for run in runs):
        parser.error("Arithmetic panel requires matching per-request FLOP models")
    if standalone:
        plt.rcParams.update({"font.size": 14, "axes.titlesize": 16})
        figure, axis = plt.subplots(figsize=(8, 5.5))
        axes = [axis]
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    if args.panel != "arithmetic":
        x = list(range(len(runs)))
        bottom = [0.0] * len(runs)
        for key, label, color in [
            ("aten::copy_", "SDPA: copies / conversions", "#4477AA"),
            ("aten::mul", "SDPA: scaling", "#66CCEE"),
            ("aten::bmm", "SDPA: attention products", "#228833"),
            ("attention_other", "SDPA: other", "#CCBB44"),
            ("outside_sdpa", "Outside SDPA", "#BBBBBB"),
        ]:
            values = [run["metrics"]["group_activity_ms"].get(key, 0.0) for run in runs]
            axes[0].bar(x, values, bottom=bottom, label=label, color=color, width=0.55)
            bottom = [a + b for a, b in zip(bottom, values)]
        for i, run in enumerate(runs):
            metrics = run["metrics"]
            axes[0].text(i, bottom[i] + 0.8,
                         f"{bottom[i]:.2f} ms\nSDPA {metrics['attention_share_of_device_activity']:.1%}",
                         ha="center", fontsize=13 if standalone else 9)
        axes[0].set_xticks(x, [f"B={run['batch_size']}" for run in runs])
        axes[0].set_ylim(0, max(bottom) * 1.25)
        axes[0].set_ylabel("Recorded device execution (ms); gaps excluded")
        axes[0].set_title("Measured: SDPA dominates execution time")
        axes[0].legend(fontsize=11 if standalone else 8, loc="upper left")
    if args.panel != "execution":
        arithmetic_axis = axes[0] if args.panel == "arithmetic" else axes[1]
        flops = runs[0]["flops_per_layer_per_request"]
        keys = ["attention_products", "qkvo_projections", "mlp_projections"]
        values = [flops[key] / 1e6 for key in keys]
        arithmetic_axis.bar(range(3), values, color=["#228833", "#AA3377", "#EE7733"])
        arithmetic_axis.set_xticks(range(3), ["Attention products\nQKᵀ + AV", "Attention projections\nQ/K/V/output", "MLP projections"])
        for i, value in enumerate(values):
            arithmetic_axis.text(i, value + 0.4, f"{value:.2f}", ha="center", fontsize=13 if standalone else 9)
        arithmetic_axis.set_ylim(0, max(values) * 1.2)
        arithmetic_axis.set_ylabel("Modeled million FLOPs per layer per request")
        arithmetic_axis.set_title("Modeled: MLP has more arithmetic")
    device_names = {run["environment"]["device_name"] for run in runs}
    contexts = {run["config"]["input_tokens_per_request"] for run in runs}
    if len(device_names) != 1 or len(contexts) != 1:
        parser.error("Chart requires matching device names and context lengths")
    if standalone:
        title = f"{'DEBUG — ' if debug else ''}{next(iter(device_names))} cached decode, N={next(iter(contexts))}"
        if args.panel == "arithmetic":
            title += "\nModeled matrix arithmetic"
            axes[0].set_ylabel("Million FLOPs per layer per request")
        else:
            axes[0].set_ylabel("Recorded device execution (ms)")
        axes[0].set_title(title)
        figure.tight_layout()
    else:
        figure.suptitle(f"{'DEBUG — ' if debug else ''}{next(iter(device_names))} cached decode, N={next(iter(contexts))}: "
                       "execution time and arithmetic work")
        figure.text(0.5, 0.02, "Diagnostic captures; time coverage is not hardware utilization. "
                    "FLOPs exclude elementwise work and vocabulary head.", ha="center", fontsize=9)
        figure.tight_layout(rect=[0, 0.055, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    provenance = {
        "summary_path": str(args.summary),
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "source_run_ids": [run["run_id"] for run in runs],
        "source_mode": "debug" if debug else "strict", "publication_eligible": not debug,
    }
    if standalone:
        scope = ("Modeled leading matrix FLOPs per layer per request; multiply-add counts as two; "
                 "excludes elementwise work and vocabulary head" if args.panel == "arithmetic" else
                 "Recorded forward device execution by operator ancestry; gaps excluded; one trace per batch")
        provenance.update({"source_mode": "debug" if debug else "strict", "publication_eligible": not debug,
                           "scope": scope})
    write_json(args.output.with_suffix(".sources.json"), provenance)
    print(f"plot_path: {args.output}")


if __name__ == "__main__":
    main()
