"""Validate and analyze a retained both/all counter session without a GPU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import platform
import sys

from dunlin.artifacts import write_json
from dunlin.attention_bandwidth import analyze_session, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="New analysis directory")
    parser.add_argument("--require-strict", action="store_true",
                        help="Reject debug sessions for official study analysis")
    args = parser.parse_args()
    try:
        summary = analyze_session(args.session, require_strict=args.require_strict)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[1]
    summary["analysis_environment"] = {"python": sys.version, "platform": platform.platform()}
    summary["analysis_source_sha256"] = {
        name: sha256(root / name) for name in
        ("experiments/analyze_attention_bandwidth.py", "src/dunlin/attention_bandwidth.py")
    }
    write_json(args.output_dir / "summary.json", summary)
    debug = summary.get("source_mode") == "debug"
    if debug:
        print("DEBUG — development analysis; not eligible for publication")
    fields = ["capture_order", "kernel", "batch_size", "sample_count", "replay_passes",
              "duration_ns", "duration_us", "dram_read_bytes", "dram_write_bytes", "dram_total_bytes",
              "dram_gb_per_second", "dram_pct", "l2_pct", "l2_hit_pct", "l1tex_pct", "sm_pct",
              "kernel_name", "grid", "block"]
    if debug:
        fields.extend(("source_mode", "publication_eligible"))
    with (args.output_dir / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{**row, "source_mode": "debug", "publication_eligible": False}
                          if debug else row for row in summary["captures"]])
    comparisons = [{**row, "source_mode": "debug", "publication_eligible": False}
                   if debug else row for row in summary["comparisons"]]
    with (args.output_dir / "comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    for row in summary["comparisons"]:
        print(f"{row['kernel']}: duration {row['duration_ratio']:.3f}x, "
              f"DRAM bytes {row['dram_bytes_ratio']:.3f}x, "
              f"bandwidth {row['dram_bandwidth_ratio']:.3f}x")
    print(f"summary_path: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
