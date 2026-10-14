# Reproduce the inference investigation

Start by rebuilding the saved-data analyses. They require no GPU, model weights,
or Nsight installation. Collecting new measurements is a separate workflow and
produces a new result under the recorded hardware, runtime, and source conditions.

Run commands from the repository root. Use new output directory names for each
analysis or capture; analyzers refuse to replace existing directories.
Saved-data analysis and plotting do not require a clean analysis checkout.

## Setup

The project requires Python 3.13 or newer. Install the environment from the
committed lockfile and inspect the resolved runtime:

```bash
uv sync --locked
uv run --locked python src/report_environment.py
```

The [reports](../README.md#first-investigation-from-batching-behavior-to-attention-costs)
describe the original T4 runtime. Installing the current lockfile reproduces the
released software environment for your platform; it does not recreate the
original host, GPU state, or capture checkout. See
[source identities in saved results](artifacts.md#source-identities-in-saved-results).

## Rebuild the prefill/decode matrix

```bash
uv run --locked python experiments/plot_batch_context_matrix.py \
  results/published/20261003T150232Z_batch_context_matrix_9d7ad269 \
  --output runs/rebuilt_t4_matrix.png

uv run --locked python experiments/plot_context_scaling.py \
  results/published/20261003T150232Z_batch_context_matrix_9d7ad269 \
  --batch-size 1 --output runs/rebuilt_t4_context_b1.png
```

The matrix plotter validates completed parent/child records and rebuilds medians
from child `measurements.csv` files. The context plot is an optional view of the
same samples. Both emit companion `.sources.json` records. Explicit `--output`
paths can overwrite an existing figure, so choose fresh paths to preserve it.
Read the [matrix report](../results/studies/prefill_decode_scaling.md) for the
single-sweep limit and the difference between prefill and decode throughput.

## Rebuild decode scaling

Keep the three series separate and supply runs in their execution order.
Each series validates workload, source, lockfile, and environment compatibility
before recalculating medians, inclusive p10/p90, throughput, and gain criteria.

```bash
uv run --locked python experiments/analyze_decode_scaling.py \
  results/published/20261009T200203Z_t4_decode_n500_b8 \
  results/published/20261009T200203Z_t4_decode_n500_b16 \
  --require-strict --output-dir runs/rebuilt_t4_anchor

uv run --locked python experiments/analyze_decode_scaling.py \
  results/published/20261009T213223Z_t4_decode_n500_repeat_b32 \
  results/published/20261009T213223Z_t4_decode_n500_repeat_b16 \
  --require-strict --output-dir runs/rebuilt_t4_reverse

uv run --locked python experiments/analyze_decode_scaling.py \
  results/published/20261009T215054Z_t4_decode_n500_b16 \
  results/published/20261009T215054Z_t4_decode_n500_b32 \
  --require-strict --output-dir runs/rebuilt_t4_forward

uv run --locked python experiments/plot_decode_scaling.py \
  runs/rebuilt_t4_forward/summary.json \
  --output runs/rebuilt_t4_forward/scaling.png
```

Use the same plot command with the anchor or reverse summary to rebuild their
figures. Do not combine these inputs into one source-matched B=8/16/32 curve or
pool their samples. The [scaling report](../results/studies/decode_batch_scaling.md)
documents the separate source commits, execution orders, and study limits.

## Rebuild trace attribution

```bash
uv run --locked python experiments/analyze_trace_attention.py \
  results/published/20261010T131445Z_t4_decode_attribution_n500_profile_b16 \
  results/published/20261010T131445Z_t4_decode_attribution_n500_profile_b32 \
  --require-strict --output-dir runs/rebuilt_t4_execution

uv run --locked python experiments/plot_trace_attention.py \
  runs/rebuilt_t4_execution/summary.json \
  --panel execution --require-strict \
  --output runs/rebuilt_t4_execution/execution.png

uv run --locked python experiments/plot_trace_attention.py \
  runs/rebuilt_t4_execution/summary.json \
  --panel arithmetic --require-strict \
  --output runs/rebuilt_t4_execution/matrix_arithmetic.png
```

Outputs include the complete analysis summary, `kernel_ranking.md`, and a variant
CSV per capture. Attribution follows owning CPU operators and interval unions;
see the [execution report](../results/studies/decode_execution.md#timing-and-attribution-method).
These diagnostic traces do not measure attention's contribution to forward
latency without profiling.

Use `--require-strict` for official trace analysis. It rejects debug captures,
requires a recorded clean capture checkout and full source commit, and validates
recorded source checks when present. Historical captures without source-mode
fields are treated as strict; this does not imply that later boundary checks
were performed. Unknown or inconsistent recorded modes are rejected.

For exploratory debug traces, omit `--require-strict`. The summary, ranking
Markdown, ranking CSVs, and supported plots carry debug labels and remain
ineligible for publication, even if the original checkout happened to be clean.
Completion, workload, attribution, and output correctness checks still apply.

## Rebuild hardware-counter analysis

```bash
uv run --locked python experiments/analyze_attention_bandwidth.py \
  results/published/20261011T153633Z_t4_attention_bandwidth_2g0yi2bn \
  --require-strict --output-dir runs/rebuilt_t4_bandwidth

uv run --locked python experiments/plot_attention_bandwidth.py \
  runs/rebuilt_t4_bandwidth/summary.json \
  --output runs/rebuilt_t4_bandwidth/attention_bandwidth.png
```

The analyzer consumes exported counter CSVs, selection/configuration records,
and all 80 replay workload records. It writes summary, measurements, and batch
comparison files; the plotter also records figure sources. The included migrated
capture configuration makes this workflow independent of historical Git source
and original host paths. Read the [bandwidth report](../results/studies/decode_attention_bandwidth.md)
for units and sampling limits, and the
[artifact guide](artifacts.md#attention-counter-captures) for the saved files.

## Rebuild the README overview

After rebuilding trace attribution and hardware-counter analysis above, generate
the two reusable plots independently:

```bash
uv run --locked python experiments/plot_trace_attention.py \
  runs/rebuilt_t4_execution/summary.json \
  --panel execution --require-strict --output runs/rebuilt_t4_execution.png

uv run --locked python experiments/plot_attention_bandwidth.py \
  runs/rebuilt_t4_bandwidth/summary.json \
  --panel dram-throughput --require-strict --output runs/rebuilt_t4_dram.png
```

Each plot uses only its own analysis summary and emits its own `.sources.json`,
recording the input path/hash, source run or session IDs, plotting-code hash,
source mode, and scope. The execution plot sums recorded device activity across
each diagnostic forward, excluding gaps. The DRAM plot shows exported throughput
percentages for individual selected launches from a separate counter capture.
Both check recorded source and correctness; the DRAM plot also checks complete
B=16/32 selections and replay records. Reuse either plot alongside other figures;
keep capture scope and interpretation limits in the accompanying text. The
execution report reuses the execution plot and displays modeled arithmetic as
its own standalone figure.

For exploratory debug inputs, omit `--require-strict`; each plot carries a
visible DEBUG label and `publication_eligible: false`. The input-validation
checks still apply.

## Source policy for new runs

By default, workload runners require a clean Git checkout, including staged,
unstaged, and untracked files. Ignored outputs and caches are allowed. Commit
experiment changes before an official run. Runners record the starting commit
and check that it and the clean state remain stable at phase/capture boundaries,
between matrix points, and before completion. These checks occur outside timing
and profiler ranges. A failed check stops the run and preserves available records.

Keep an isolated checkout unchanged during long captures: boundary checks cannot
detect every edit followed by a revert between checks.

Use `--debug` to develop experiments before committing. It records source state
while allowing local changes; workload and correctness validation still apply.
Debug results are for exploration and remain ineligible for official comparisons
even if the checkout happened to be clean. Scaling, trace, and bandwidth analyses
label debug outputs with `publication_eligible: false`; supported plots carry
visible debug labels. Use `--require-strict` for official analysis of each study.

## Run new CPU, MPS, or CUDA benchmarks

The first workload invocation downloads the pinned model/tokenizer when they
are not cached. A short CPU smoke run checks execution and artifact creation:

```bash
uv run --locked python experiments/batched_forward_phases.py \
  --config configs/smoke_forward_phases.toml
```

The smoke preset uses float32, N=32, B=1, one warm-up, and two measured forwards
per phase. It is an execution check, with a different protocol from the T4 study.
Benchmark points check decode cache lengths; profile runs add output-reference
checks. A successful timing run alone does not certify full output agreement.

Preview the full matrix without loading a model:

```bash
uv run --locked python experiments/batch_context_matrix.py \
  --config configs/batch_context.toml --device cpu --dtype float32 --dry-run
```

Remove `--dry-run` to execute. On a Mac with MPS available, use `--device mps
--dtype float16`; for CUDA, use `--device cuda:0 --dtype float16`. Requested
unavailable devices fail rather than falling back, including in benchmark/profile
dry runs. These previews resolve settings and check device availability without
loading model weights. The full preset pins Qwen2.5-0.5B-Instruct and its
tokenizer to `7ae557604adf67be50417f59c2c2f167def9a775`, with five warm-ups
and 30 samples per phase/point. Backend or precision changes define different
conditions and must retain their labels.

The current forward workload supports Qwen2 causal-LM checkpoints. Other
architectures require a compatible runner; a different checkpoint is a different
workload even when its token counts match.

Runners print the created run directory. The matrix starts a fresh child process
per point and stops at a failure, preserving partial artifacts. Plot a completed
new matrix with `plot_batch_context_matrix.py RUN_DIRECTORY --output FIGURE.png`.

For forward/matrix runs, `--output-dir` chooses an output root; otherwise use
`DUNLIN_RESULTS`, the legacy `INFERENCE_LAB_RESULTS` fallback, or local `runs/`.
`--run-dir` selects an exact directory for a forward or profile point and cannot
be combined with `--output-dir`. CLI settings override the TOML preset. Each run
records requested/resolved settings and copies the supplied preset when present.

## Collect new matched CUDA traces

Use a compatible CUDA host. The following produces a new N=500, B=16/32
decode-only timing pair; replace `my_t4_decode` with a fresh label before reruns.
Inspect completion and environment records after each point and record execution
order, available memory, elapsed budget, failures, and stopping decisions.

```bash
prefix="runs/my_t4_decode"

uv run --locked python experiments/batched_forward_phases.py \
  --config configs/batch_context.toml --phase cached_decode \
  --input-tokens 500 --batch-size 16 --device cuda:0 --dtype float16 \
  --run-dir "${prefix}_b16"

uv run --locked python experiments/batched_forward_phases.py \
  --config configs/batch_context.toml --phase cached_decode \
  --input-tokens 500 --batch-size 32 --device cuda:0 --dtype float16 \
  --run-dir "${prefix}_b32"

uv run --locked python experiments/profile_forward_phases.py \
  --baseline-run-dir "${prefix}_b16" --phase cached_decode \
  --run-dir "${prefix}_profile_b16"

uv run --locked python experiments/profile_forward_phases.py \
  --baseline-run-dir "${prefix}_b32" --phase cached_decode \
  --run-dir "${prefix}_profile_b32"
```

Baseline mode takes the completed timing run's effective workload and checks
loaded input identity. Do not combine it with manual workload flags. The profile
runner initializes profiling with a discarded capture, then saves a diagnostic
forward and validates finite logits, shape, fresh-cache reference agreement,
cache lengths, and CUDA activity. `validation.json` records the checks.
Default traces omit shape/stack/memory recording; `--record-shapes` is a separate
diagnostic condition. On the capture host, preview a profile command with
`--dry-run` before collection; its selected CUDA device must be available.

The timer includes synchronized forward execution and cache updates. Model
loading, tokenization, transfer, and fresh prompt-cache preparation are outside
cached-decode timing. Prefill throughput is B×N/median seconds; one-token decode
throughput is B/median seconds. These are isolated forward metrics, with no
serving queue or full autoregressive generation.

For a new scaling study, declare criteria before measurement. The retained study
used gain below 25% with rising latency as weakened scaling, and required gains
below 10% on two successive doublings plus replication for a plateau. Repeat a
candidate pair in reverse order and analyze each pair separately. The commands
above supply a pair, rather than automatically carrying out that full protocol.

## Collect new attention counters

This runner's five selectors are specific to the validated T4 B=16/32, N=500
execution path. On a compatible native Linux CUDA host, install Nsight Compute
CLI, ensure `ncu` and `nvidia-smi` are available, and arrange hardware-counter
access with the host configuration. Use matching recent baselines and traces.
Changed names, grids, ordering, runtime, or workload require investigating the
selection before interpreting a counter report.

Preview using the new timing prefix above; the preview needs no GPU or Nsight:

```bash
uv run --locked python experiments/profile_attention_bandwidth.py \
  --baseline-prefix "${prefix}" --batch both --kernel all --dry-run
```

On the capture host, check selections in a separate LaunchStats-only session:

```bash
uv run --locked python experiments/profile_attention_bandwidth.py \
  --baseline-prefix "${prefix}" --batch both --kernel all --preflight
```

Preflight does not measure bandwidth. If selection and workload validation pass,
collect the complete counter session:

```bash
uv run --locked python experiments/profile_attention_bandwidth.py \
  --baseline-prefix "${prefix}" --batch both --kernel all
```

Each invocation prints a unique session directory. Analyze the complete counter
session, using that directory instead of the published session in the saved-data
command above. The bandwidth analyzer requires all five selections at both
batches; a single-kernel, preflight, or details-only session is not a substitute.
The default capture command points to the original published baselines; this
guide supplies fresh ones explicitly.

The runner validates each replay and selection, records configuration before
collection, retains commands/tool/GPU metadata, and stops at the first failure.
It leaves caches and clocks uncontrolled. Save native `.ncu-rep` reports and
partial/failed directories along with CSVs before discarding a capture host.
Set and record a collection budget before starting. Recollection should preserve
the actual outcome rather than assuming the original numbers will repeat.
