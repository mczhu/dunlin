# Results

Read the T4 investigation in order:

1. [Prefill and decode scaling](studies/prefill_decode_scaling.md)
2. [Decode batch scaling](studies/decode_batch_scaling.md)
3. [Decode execution](studies/decode_execution.md)
4. [Attention bandwidth](studies/decode_attention_bandwidth.md)

## Evidence layout

- `published/`: selected capture samples, configurations, metadata, and validation
  records in their original run directories.
- `studies/`: reports, derived analyses, and figures.
- `../runs/`: ignored outputs from new measurements and analysis.

Use the [reproduction guide](../docs/reproduce.md) to rebuild results or collect
new measurements, and the [artifact guide](../docs/artifacts.md) to interpret the
saved records. The [project README](../README.md) presents the findings and
research direction.
