# AGENTS.md

## Project

Dunlin builds reproducible experiments to measure transformer inference
performance, explain its limits, and evaluate optimizations. The current release
studies dense inference on one GPU. Serving, distributed execution, and sparse
architectures are planned research directions.

## Priorities

1. Correctness before optimization.
2. Reproducibility before convenience.
3. Simple, inspectable code over abstractions.
4. Measure before explaining.
5. Form hypotheses before changing performance-sensitive code.

## Coding and experiments

- Use Python and type hints where they improve clarity.
- Keep measurement separate from analysis and plotting.
- Keep workload device selection configurable for CPU, MPS, and CUDA. State the
  NVIDIA hardware/tool requirements of CUDA-specific diagnostics.
- Record effective workload settings, input identity, environment/hardware,
  source, and validation with results.
- Test metric calculations and non-performance-sensitive utilities.
- Keep dependencies minimal; introduce abstractions only for a concrete need.
- Declare competing predictions, timing boundaries, comparison controls, and
  stopping criteria before a study. Preserve failures and protocol deviations.
- Distinguish logical work models, benchmark samples, diagnostic traces, and
  hardware counters. A resource explanation needs supporting measurements;
  causal performance attribution needs a controlled intervention.
- Preserve original measurements and their validation levels. Put new reports
  in `results/studies/`, with links to supporting runs under `results/published/`.

## Experiment source provenance

- Every workload runner and sweep must use the shared `dunlin.source` checks.
- By default, require a clean Git checkout, including staged, unstaged, and
  untracked files, before loading a workload or launching a child benchmark.
  Ignored files are allowed.
- Pin the starting commit and check cleanliness and commit identity at
  phase/capture boundaries, between sweep points, and before reporting completion.
  Keep Git calls outside timed intervals and profiler ranges. A failed check
  must fail the run.
- Keep check records with artifacts, or on stderr for stdout-only runners.
- Offer explicit `--debug` for development before committing. Record source state
  without enforcing cleanliness or commit stability, propagate the mode to
  children, and label artifacts and plots. Reject debug results from publication
  and official study comparisons even when the checkout happened to be clean.
  Retain workload and correctness validation in debug mode.
- Exploratory analysis and plots of debug inputs must have a visible debug label
  and `publication_eligible: false`. Official analysis must explicitly require
  strict source. Retain workload, sample, and correctness checks in both paths.
- Dry runs, environment diagnostics, analysis, and plotting of saved data are
  exempt from requiring a clean current checkout. This does not waive validation
  of their recorded inputs.
- Boundary checks cannot detect every edit followed by a revert. Use an isolated
  checkout for long captures and keep it unchanged until completion.

Use [reproduction](docs/reproduce.md) for commands and the
[artifact guide](docs/artifacts.md) for record meanings and historical provenance.
Use `--require-strict` for official scaling, trace, and bandwidth analysis.
Historical captures without source-mode fields remain legacy inputs; current
checks must not be retroactively attributed to them.

## Documentation math on GitHub

- Use fenced `math` blocks for multiline display equations. Keep TeX line breaks,
  subscripts, alignment markers, and braces inside the fence so Markdown does
  not interpret them. Do not add `$$` delimiters inside a `math` fence.
- Use simple inline math; use GitHub's dollar-and-backtick delimiters when TeX
  overlaps with Markdown syntax.
- Avoid `\operatorname` in GitHub math: it was rejected by the repository preview.
  Use `\mathrm` for upright labels, or explain labels in prose.
- When changing equations, inspect every math block in the affected document and
  check the GitHub preview before publication. A generic Markdown preview or CI
  pass does not establish that GitHub rendered the math correctly. Preserve
  equations and measured values when making formatting fixes.
- Publication from the private preparation repository enforces a deterministic
  check of these math rules before export. Retain the GitHub visual review.
- Follow [GitHub's math formatting guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
