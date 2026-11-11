# Reproducing the study

> This page documents the earlier v0.1.0 quadratic pipeline. For the current analysis use the [spline guide](SPLINES.md). `make reproduce` now redraws spline figures without fitting; use `make legacy-reproduce` for the pipeline below.

## Environment

- Python **3.11** (`pyproject.toml` requires >=3.11,<3.12); exact dependencies are in `uv.lock`.
- R and the pinned packages in `requirements.R`. Packages install into the project-local `.Rlib/` directory. R itself and transitive R dependencies are not fully locked.
- A working C/C++ compiler for PyTensor model compilation. `analysis/runtime.py` contains the macOS compiler compatibility adjustment used by this project.
- GNU Make, or run the Python/R commands directly below.

The retained analysis environment used Python 3.11.13 on macOS. A fresh installation on every operating system has not been verified. The current full working files include approximately 209 MB of data and 7.3 GB of results, principally posterior draws. Allow extra disk space for Git history, dependencies, compilation caches, and refits. Bayesian fits normally use four chains and four worker processes.

The GitHub repository stores the source materials and readable outputs. Posterior draws are separate release assets. Download the eight model ZIPs from [release v0.1.0](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996/releases/tag/v0.1.0), then run `python3 script/restore_models.py --archives /path/to/downloads`. Downloaded ZIPs require roughly 7.3 GB in addition to the extracted models; they may be removed after successful verification. Run `python3 script/restore_models.py --check` to verify existing restored models without downloading anything.

## Commands

Run commands from the repository root.

| Purpose | Command |
| :--- | :--- |
| Install dependencies | `make setup` |
| Verify retained inputs and sample counts | `make check` |
| Run small offline regression checks | `make test` |
| Try the standard-library example | `make example` |
| Regenerate analyses, reusing saved draws | `make legacy-reproduce` |
| Refit all Bayesian models and regenerate outputs | `make refit` |
| Export submission statistics and article tables from existing results | `make tables` |
| Export descriptive statistics and quadratic turning-point draws | `make submission` |

Without Make:

```sh
uv sync --locked
R_LIBS_USER="$PWD/.Rlib" Rscript --vanilla requirements.R
uv run --locked python check.py
uv run --locked python -m unittest discover -s tests -v
uv run --locked python reproduce.py --legacy
```

`reproduce.py` configures project-local plotting/compiler caches and the R library path. It calls `requirements.R` to check/install the required R packages before running. Once dependencies are installed, the scientific steps use local files. Annotation credentials are not required.

## Execution order

The complete runner performs:

1. Input checks and French source-length metadata preparation.
2. Human/model validation estimates.
3. Primary ordinal models, supplementary signalling models, and Babel models.
4. Two-lines tests and Holm adjustment.
5. French source-length sensitivity and Babel contrasts.
6. Observed occurrence/conditional-importance decomposition and later-period models.
7. Temporal histograms, submission statistics, and manuscript table/figure indexes.

`results/reproduction.json` records each step's exit status and elapsed time. A nonzero status stops the run. Inspect the final entry for the failing step. A completed saved report is evidence of that recorded run, not a substitute for testing a new environment.

## Saved draws versus refitting

The default run loads an existing `.nc` posterior file when one is present; missing models are fitted. It is therefore not a guaranteed fast mode if draws are absent. `--refit` forces fresh estimation.

**After changing data, formulas, priors, or model settings, run `make refit`.** Cache filenames do not encode those changes. Do not mix new data with old posterior files. For a new study, use a separate copy and output location.

Model sampling and bootstrap seeds are set in the scripts. Frozen annotations preserve the original model scores; generating new annotations is a separate operation and is not expected to reproduce them exactly. Different compilers or sampling libraries can also change posterior draws despite identical seeds.

## Targeted stages

`make prepare`, `make validation`, `make primary`, `make signaling`, `make latent`, `make twolines`, `make length`, `make descriptive`, and `make two-part` expose individual stages. `make figures` redraws the temporal histograms and decomposition panels from existing analysis outputs. It is not a replacement for the full runner.

`make tables` requires the earlier analysis outputs and saved models. It does not run missing upstream analyses. For a first reproduction, use `make legacy-reproduce`.

## What can and cannot be reproduced

Included: exact retained primary annotation inputs, text hashes, frozen scores, bibliography/author metadata, annotation definitions, human-validation records, model code, posterior draws, figures, and numerical outputs.

Not fully recoverable from the release: the original upstream acquisition and random passage-selection process, all provider access dates, raw API responses, retry counts, and complete run provenance for every annotation batch. The exact retained passages can be verified, but reconstructing the original sample from every upstream source is a different task. `analysis/prepare.py` prepares retained French length metadata; it does not resample the original corpus.

For public-release checks and storage packaging, see [RELEASE.md](RELEASE.md).
