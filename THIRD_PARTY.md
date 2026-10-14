# Third-party components

Dunlin uses external libraries, model files, and development tools. Their
licenses and notices remain separate from the license for Dunlin's original
material. This repository contains experiment code and selected generated
evidence; it does not distribute library binaries, model/tokenizer files,
NVIDIA tools, or native Nsight reports.

## Runtime dependencies

The [lockfile](uv.lock) records the installed dependency versions. Direct
dependencies declared in `pyproject.toml` are:

| Component | Upstream project license | Use |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style project license; packaged components have additional license terms | Tensor operations, model execution, and profiling |
| [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) | Apache-2.0 | Model/tokenizer loading and Qwen model implementation |
| [Safetensors](https://github.com/safetensors/safetensors/blob/main/LICENSE) | Apache-2.0 | Model-weight loading |
| [Matplotlib](https://matplotlib.org/stable/project/license.html) | Matplotlib license, based on the PSF license | Figures |
| [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE) | BSD-3-Clause project license | Declared tabular-data dependency |

These links identify the upstream projects. Installed distributions may include
additional component licenses and notices, including those of transitive
dependencies. Preserve the license files supplied with the particular
distributions when redistributing them. This summary does not replace their
packaged terms.

## Model and tokenizer

The published workload uses `Qwen/Qwen2.5-0.5B-Instruct`, with model and
tokenizer pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`.
That revision carries the [Apache-2.0 license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE).
The runners download these files separately when needed; they are not included
in this release. The terms of that checkpoint are specific to it and do not
describe every model in the Qwen family.

## Diagnostic and development tools

CUDA and Nsight Compute are NVIDIA products with their own terms. The retained
captures record CUDA build 13.0 and Nsight Compute 2026.1.0.0. See the
[CUDA 13.0 agreement](https://docs.nvidia.com/cuda/archive/13.0.0/eula/index.html)
and [Nsight Compute 2026.1 license](https://archive.docs.nvidia.com/nsight-compute/2026.1/CopyrightAndLicenses/index.html).
The release includes exported measurements and capture metadata, rather than
redistributing those tools.

The trace illustration shows [Perfetto](https://github.com/google/perfetto/blob/main/LICENSE),
which uses Apache-2.0 for its main code with separate notices for listed
components. The screenshot identifies the viewer used to inspect the generated
trace; it does not transfer ownership of the viewer's interface or trademarks.

Python, Git, uv, build tools, and GitHub Actions are obtained separately. Consult
their distributions for applicable terms. Product names, kernel identifiers,
and tool interfaces in reports identify the measurement environment; they do
not imply endorsement by their owners.
