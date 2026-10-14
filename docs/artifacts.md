# Experiment artifacts

This guide explains the files produced by timing runs, matrix sweeps, forward
traces, attention counter captures, and their saved-data analyses. Use it to find
the workload settings, measurements, correctness checks, and analysis outputs
behind a result. For commands, see the [reproduction guide](reproduce.md).

New runs create a separate directory under `runs/` and print its path. `runs/`
is Git-ignored. Published evidence lives under `results/published/`.

Saved-data analysis reads retained files without running inference or retrieving
the original source checkout. Recorded host paths are historical metadata and
need not exist on the analysis machine. The required saved inputs must still be
present and pass validation.

- [What the release includes](#what-the-release-includes)
- [Shared run records](#shared-run-records)
- [Source identities in saved results](#source-identities-in-saved-results)
- [Checksums and published files](#checksums-and-published-files)
- [Timing runs and matrix sweeps](#timing-runs-and-matrix-sweeps)
- [Forward traces](#forward-traces)
- [Analysis summaries and plots](#analysis-summaries-and-plots)
- [Attention counter captures](#attention-counter-captures)

## What the release includes

| Study | Included evidence |
| --- | --- |
| [Prefill/decode matrix](../results/studies/prefill_decode_scaling.md) | One sweep of 20 workload points, with 30 timing samples per phase/point |
| [Decode scaling](../results/studies/decode_batch_scaling.md) | Six timing runs: a B=8/16 anchor and two separate B=16/32 comparisons, plus a supporting B=8 correctness check |
| [Matched execution](../results/studies/decode_execution.md) | Two complete forward traces, at B=16 and B=32 |
| [Attention counters](../results/studies/decode_attention_bandwidth.md) | Ten selected launches across B=16/32, with 80 validated application replays |

The linked reports explain each study's methods and interpretation limits. Their
analyses and figures can be rebuilt from the retained inputs. FLOP and traffic
estimates use architecture and workload dimensions; trace durations and DRAM
counters are measured evidence. MLP hardware counters were not collected.

Some per-run summaries, logs, ranking snapshots, replay presets, and diagnostic
views are omitted because the documented analyses do not need them. Manifests
can therefore list files that are absent from this repository. The supporting
B=8 diagnostic retains its correctness records; its trace is omitted and it is
excluded from timing. Native Nsight `.ncu-rep` reports are Git-ignored; the
counter analyzer uses the included CSV exports.

## Shared run records

Timing and profiling runners use the following files. Which records are present
depends on the runner and capture mode.

| File | Contents |
| --- | --- |
| `manifest.json` | Run ID, experiment and methodology version, start/end times, completion status, failure details, and applicable source mode and linked baseline. Matrix manifests also list child points. |
| `config.resolved.json` | Effective workload: model/tokenizer revisions, input identity, batch and token counts, phase, device/dtype, cache and attention settings, and applicable warm-ups, repetitions, metric definitions, and validation tolerances. |
| `config.source.toml` | Copy of the supplied preset, when used. The resolved configuration includes any command-line overrides. |
| `environment.json` | Recorded source commit and checkout state, lockfile hash, software versions, OS, backend, hardware, and applicable memory/driver details. |
| `validation.json` | Profiling correctness checks: logits shape/finiteness, reference agreement, cache lengths, and applicable profiler activity checks. Ordinary timing runs do not produce this record. |
| `source-checks.jsonl` | Source checks recorded by current runners at execution boundaries. Historical runs may predate these checks. |
| `stdout.log` | Runner diagnostics and printed results, where produced. Structured measurements and JSON records are the analysis inputs. |

Start with the manifest: `complete` means the runner finished; `failed` includes
a reason; `running` can indicate an interrupted run. A completed timing run does
not imply that profiling reference checks were performed. Read the configuration,
environment, measurements, and any validation record alongside it to assess
workload identity and the checks actually performed. Output and cache checks
apply within the recorded tolerances; they do not establish performance stability
or explain a hardware bottleneck. Older results may lack fields that current
runners produce; use the fields actually recorded for each run.

For examples, inspect a retained timing run's
[`manifest.json`](../results/published/20261009T215054Z_t4_decode_n500_b16/manifest.json),
[`config.resolved.json`](../results/published/20261009T215054Z_t4_decode_n500_b16/config.resolved.json),
and [`environment.json`](../results/published/20261009T215054Z_t4_decode_n500_b16/environment.json).

## Source identities in saved results

The commit IDs and lockfile hashes in saved results identify the code and
dependencies used during capture. The original capture commits belong to an
earlier source history and are not available to check out in this repository.
You do not need them to analyze the saved data: the workload settings and
measurements are included as files.

Recorded machine paths in the published copies use `/capture/project/` or
`/capture/python/` placeholders. These identify historical locations; they are
not paths to create on your machine. Trace hostnames and GPU snapshot identifiers
also use consistent placeholders. Timing samples, counter values, and trace
events retain their original values.

Use the released code to run the documented workflows and compare your results
under the conditions recorded by your run. This code includes source checks and
artifact support added after the original experiments. An older clean-source
record does not imply that those later checks were performed. Repeating an
experiment with current code also does not establish execution of the exact
historical implementation.

The [historical attention configuration](#historical-configuration) is marked as
reconstructed after capture; its measured counters and replay validations are
unchanged.

## Checksums and published files

Hashes in analysis summaries and figure source records identify the files used
to produce them. Rebuilt analyses hash the published input files; historical
records may identify the original files from capture.

If a file was changed for publication, for example to replace a host path, its
entry in [`publication/redactions.json`](../publication/redactions.json) connects
the original and published versions:

| Field | Meaning |
| --- | --- |
| `destination` | File path in this repository |
| `original_sha256` | SHA-256 hash of the original file |
| `exported_sha256` | SHA-256 hash of the published file |
| `description` | Explanation of the change |

Unchanged files have no entry. Hashes let you check file identity and detect
changes; they do not by themselves prove that a measurement is correct or a
performance explanation is valid.

The published matrix figure source record and decode analysis summaries include
supplemental input/code hashes added during publication preparation. Their
`publication_annotation` distinguishes those records from the normal plotter or
analyzer output. These hashes describe the published inputs, not an additional
measurement.

## Timing runs and matrix sweeps

[`batched_forward_phases.py`](../experiments/batched_forward_phases.py) saves one
workload point. In addition to the shared records, it writes:

| File | Contents |
| --- | --- |
| `measurements.csv` | One row per measured repetition and phase, excluding warm-ups: run ID, repetition index, batch/input/output sizes, latency in seconds, and cached-decode cache lengths. |
| `summary.json` | Sample counts, median phase latencies, and aggregate prefill/decode throughput calculated from those samples. Metric definitions are recorded in the resolved configuration for current runs. |

See the retained
[`measurements.csv`](../results/published/20261009T215054Z_t4_decode_n500_b16/measurements.csv)
for an example of individual timing samples.

[`batch_context_matrix.py`](../experiments/batch_context_matrix.py) creates a
parent directory with one timing run per context-length/batch-size point:

```text
<matrix-run>/
├── manifest.json
├── config.resolved.json
├── environment.json
├── summary.json
├── n32_b1/
│   ├── manifest.json
│   ├── config.resolved.json
│   ├── environment.json
│   ├── measurements.csv
│   └── summary.json
└── ...
```

This layout shows the core records; presets, logs, and source checks are written
where applicable. The parent manifest lists child IDs and statuses in execution
order. Its summary contains each point's median latency and throughput speedup
relative to batch one at the same context length. Samples live in the child
directories. A partial sweep can contain completed children while its parent
remains incomplete.

Current parent manifests and summaries use `matrix_points` with schema version
2; older ones use `points` with version 1. See the included
[`matrix manifest`](../results/published/20261003T150232Z_batch_context_matrix_9d7ad269/manifest.json)
for an example. Saved-data plots rebuild results from the child samples.

## Forward traces

[`profile_forward_phases.py`](../experiments/profile_forward_phases.py) saves the
shared records and, with the PyTorch profiler enabled, these additional files:

| File | Contents |
| --- | --- |
| `trace.json` | Chrome-format execution timeline with CPU operators, GPU events, and correlation information used by trace attribution. |
| `operator_table_device.txt` | Operator aggregates sorted by self device time. |
| `operator_table_cpu.txt` | Operator aggregates sorted by self CPU time. |

The public evidence retains the required traces and shared records; the operator
tables are omitted. The reproduction guide links the
[matched B=16/32 trace inputs](reproduce.md#rebuild-trace-attribution).

The validation record checks finite `[B,1,V]` logits, agreement with a separate
reference within the recorded tolerances, and cache lengths. In PyTorch-profiler
mode, `profiled_matches_unprofiled` is true and `cuda_kernel_events` must be
positive. With `--no-torch-profiler`, both fields are null; the runner still
records `forward_matches_reference`. That mode produces no PyTorch trace or
operator tables and is used by the Nsight counter captures below.

## Analysis summaries and plots

Analysis reads the saved inputs listed below and writes a separate output
directory. A filename such as `summary.json` has a different layout for each
analysis; use its `kind` and `schema_version` to identify it.

| Analysis | Input files | Output files and contents |
| --- | --- | --- |
| Decode scaling | Each timing run's `manifest.json`, `config.resolved.json`, `environment.json`, and `measurements.csv` | `summary.json`: run identities, samples, medians, p10/p90 spread, throughput, adjacent batch-doubling gains, and compatibility/provenance records |
| Trace attribution | Each profiling run's `manifest.json`, `config.resolved.json`, `environment.json`, `validation.json`, and `trace.json` | `summary.json`: execution attribution, kernel rankings, workload/environment records, and input/analysis-code hashes. `kernel_ranking_b<B>.csv` and `kernel_ranking.md`: launch counts, durations, shares, and ownership evidence. Repeated batch captures add a `_capture<N>` suffix to CSV names. |
| Attention bandwidth | [Counter session inputs](#attention-analysis-input-and-output-artifacts) | `summary.json`, `measurements.csv`, and `comparisons.csv`, detailed below |

The plotters read their corresponding analysis summaries; matrix plotters read
the parent and child run records directly. They write:

| File | Contents |
| --- | --- |
| `<figure>.png` | Figure generated from the saved measurements or analysis. Supported debug plots carry a visible debug label. |
| `<figure>.sources.json` | Source run/session IDs and applicable input/summary hashes, plotting-code identity, tool versions, and scope. The README's standalone execution and DRAM plots each record their own summary path/hash and capture scope. Fields vary by plotter. |

Debug scaling, trace, and bandwidth analyses record `source_mode: debug` and
`publication_eligible: false`. Debug trace rankings also label Markdown and CSV
outputs. Official analysis requirements are covered in
[reproduction](reproduce.md#source-policy-for-new-runs).

## Attention counter captures

[`profile_attention_bandwidth.py`](../experiments/profile_attention_bandwidth.py)
saves Nsight counters for the T4 attention experiment at cache length 500 tokens
(N=500) and batch sizes 16 and 32 (B=16/32). Its configuration and selection
format is specific to this experiment.

### Capture configuration v1

`capture-config.json` records what the capture is intended to measure: the
workload, profiler settings, and kernel launches to select. The capture runner
uses it to build the Nsight commands; the saved-data analyzer uses it to check
that the recorded kernels and settings match those expectations. Keeping this
record lets you interpret an older capture without relying on the current
runner's defaults.

For a new collection, the runner writes the file before invoking NVIDIA Nsight
Compute, with `schema_version: 1` and `kind: attention_bandwidth_capture`. It
checks that the saved configuration stays unchanged before and after captures
and at completion, including in debug mode. Checks of the Git source state run
separately. Dry runs resolve settings in memory and write no capture artifacts.

| Field | Meaning |
| --- | --- |
| `source_mode` | Whether collection used `strict` or `debug` source checks; must match the session and replay records |
| `record_origin` | When the configuration was recorded: `capture_startup` for new collection or `legacy_migration` for a historical reconstruction |
| `preflight` | Whether this is only a kernel selection check using LaunchStats; a preflight does not measure bandwidth |
| `capture` | How Nsight collects counters: device, NVTX/range filter, kernel-name representation, launch count, replay/cache/clock settings, sections, and metrics |
| `workload` | What runs during capture: runner/interpreter, phase, profiler mode, and baseline references for each batch size |
| `points` | Which launches to measure, in order: kernel label, batch size, selector regex, launch skip, and expected grid |
| `migration` | For reconstructed configurations only: original source commit/path/hash, reconstruction time/tool hash/method, and input hashes; published copies distinguish original input hashes from hashes of transformed metadata |

Each entry in `points` describes an expected selection. For example, this entry
requests the `qk` kernel at batch size 16, using a kernel-name regex, skipping no
matching launches, and expecting the CUDA launch grid `[63, 1, 224]`. The
observed selection is saved separately in a `.selection.json` file.

```json
{
  "kernel": "qk",
  "batch_size": 16,
  "selector_regex": "gemv2T_kernel_val",
  "launch_skip": 0,
  "expected_grid": [63, 1, 224]
}
```

Supported v1 semantics select one matching launch in the second
`cached_decode_forward/` NVTX range, using demangled names and application replay,
with `cache_control: none` and `clock_control: none`. The capture runner accepts
subsets, preflight, and additional detailed sections. The bandwidth analyzer
requires the complete non-preflight session in kernel order `qk`, `kv_expand`,
`scale`, `av`, `kv_convert`, each at B=16 then B=32. A partial or detailed-only
session cannot substitute for that input contract. Unsupported configuration
fails explicitly rather than being reinterpreted using current defaults.

Selectors are specific to the recorded execution path. Name/grid agreement
alone cannot establish layer or tensor identity if launch order changes. The
[operation mapping](../results/studies/decode_attention_bandwidth.md#operation-mapping)
explains launch skips and inferred K/V roles; operand shapes/dtypes were not
recorded in the original session.

### Session and replay records

A session is one collection run. Nsight application replay reruns the workload
to collect different counters for the selected launch; each replay has its own
workload and correctness records.

For a selection such as `qk_b16` (`qk` at batch size 16), new collection writes:

| File | Role |
| --- | --- |
| `qk_b16.ncu-rep` | Native Nsight report; Git-ignored and not needed for saved-data analysis |
| `qk_b16.csv` | Measurements used to calculate bandwidth: exported names, grid, units, duration, traffic, throughput metrics, and replay pass count |
| `qk_b16.log` | Collection output for diagnosing warnings and failures |
| `qk_b16.selection.json` | Evidence of which kernel was selected: observed name/ID/grid, validation status, and validated replay workload references |
| `qk_b16_workloads/<run-id>/` | Records of the workload, environment, and output correctness for each application replay |

To inspect the retained measurements, open
[`qk_b16.csv`](../results/published/20261011T153633Z_t4_attention_bandwidth_2g0yi2bn/qk_b16.csv).
The corresponding
[`qk_b16.selection.json`](../results/published/20261011T153633Z_t4_attention_bandwidth_2g0yi2bn/qk_b16.selection.json)
shows which kernel was selected.

Session-wide records include the actual invocation and commands, completion status
and timestamps, source check records, `uv.lock`, tool version, and GPU snapshots.
These document how the collection ran and complement the configuration described
above. For which records are included in the public evidence, see
[the release overview](#what-the-release-includes).

Every replay retains the shared `manifest.json`, `config.resolved.json`,
`environment.json`, and `validation.json` records. The analyzer checks completion,
source mode, workload/environment agreement, and output correctness: finite
`[B,1,151936]` logits, fresh-cache reference agreement, and cache length 500→501.
It also compares the kernel names and grids in the CSV and selection records
with the configuration's expectations. Native Nsight selection checks supply
the kernel evidence; these replays use `--no-torch-profiler` as described above.

The retained session has ten reports with eight replay passes each: 80 validated
workloads. Replay combines different metrics for one selected launch. It does
not supply 80 independent performance samples or a variability estimate.

### Attention analysis input and output artifacts

The analyzer uses the [configuration](#capture-configuration-v1) and
[session and replay records](#session-and-replay-records) described above:

- `capture-config.json`.
- For each selection (such as `qk_b16`): `<selection>.csv`,
  `<selection>.selection.json`, and `<selection>.log`.
- In each `<selection>_workloads/<run-id>/`: `manifest.json`,
  `config.resolved.json`, `environment.json`, and `validation.json`.
- Session-wide: `status.txt`, `settings.txt`, `invocation.sh`, `commands.sh`,
  `git-commit.txt`, `git-status.txt`, `uv.lock`, `started-at.txt`, `finished-at.txt`,
  `ncu-version.txt`, `gpu-before.txt`, and `gpu-after.txt`.
- When present: `source-mode.txt` and `source-checks.jsonl`.

The analyzer writes the following files into a new analysis directory. The
separate plotter reads `summary.json` and writes the PNG and source record
[described above](#analysis-summaries-and-plots).

| Output file | Contents |
| --- | --- |
| `summary.json` | Full analysis (`attention_bandwidth_analysis`, schema version 2): capture configuration/metadata, environment, replay checks, measurements, comparisons, metric definitions, and input/analysis-code hashes. Debug summaries include `source_mode: debug` and `publication_eligible: false`. |
| `measurements.csv` | One row per kernel/batch: duration, DRAM bytes, achieved bandwidth, throughput percentages, L2 hit rate, launch details, and replay pass count. Debug rows include source mode and publication eligibility. |
| `comparisons.csv` | B=32 / B=16 ratios of duration, DRAM traffic, and bandwidth for each kernel. Debug rows include source mode and publication eligibility. |

### Historical configuration

The [included October 11 configuration](../results/published/20261011T153633Z_t4_attention_bandwidth_2g0yi2bn/capture-config.json)
was reconstructed after collection because the session predates this file format.
It is marked `record_origin: legacy_migration`; its `migration` block records
the reconstruction's source, method, and hashes. The measured counters and replay
validation records are unchanged.

Use this file as a normal analysis input. You do not need to run a migration or
retrieve historical source. The analyzer requires `capture-config.json` and
checks the migration's input hashes against the retained files.

In the published copy, `migration.original_input_sha256` preserves the hashes
recorded by the original reconstruction. `migration.input_sha256` identifies the
published input bytes, including transformed commands and settings. The
`publication_annotation` and transformation record explain that distinction;
the original capture and reconstruction were not rerun.
