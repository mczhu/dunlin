"""Analyze saved decode points; never load a model or run inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from dunlin.artifacts import write_json
from dunlin.decode_scaling import analyze_decode_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="One independent run per batch, in execution order")
    parser.add_argument("--output-dir", type=Path, required=True, help="New analysis directory")
    parser.add_argument("--require-strict", action="store_true",
                        help="Reject debug runs for official matched scaling analysis")
    args = parser.parse_args()
    try:
        summary = analyze_decode_runs(args.run_dirs, require_strict=args.require_strict)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    write_json(args.output_dir / "summary.json", summary)
    if summary.get("source_mode") == "debug":
        print("DEBUG — development analysis; not eligible for publication")
    for point in summary["points"]:
        print(f"B={point['batch_size']}: {point['median_seconds'] * 1000:.3f} ms, "
              f"{point['tokens_per_second']:.2f} tokens/s ({point['run_id']})")
    for interval in summary["adjacent_doublings"]:
        print(f"B={interval['lower_batch']} -> {interval['upper_batch']}: "
              f"gain={interval['throughput_gain_fraction']:.1%}, "
              f"candidate={interval['weakened_scaling_candidate']}")
    print(f"summary_path: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
