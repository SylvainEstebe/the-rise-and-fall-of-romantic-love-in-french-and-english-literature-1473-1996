# Earlier quadratic analyses

[Current spline analysis](SPLINES.md)

Requirements: Python 3.11, R, a C/C++ compiler and the dependencies in `uv.lock` and `requirements.R`. Transitive R dependencies are not fully locked.

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

Models: [v0.1.0](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996/releases/tag/v0.1.0).

```sh
uv sync --locked
R_LIBS_USER="$PWD/.Rlib" Rscript --vanilla requirements.R
uv run --locked python script/restore_models.py --archives /path/to/downloads
uv run --locked python reproduce.py --legacy
```

The legacy pipeline reuses saved models and fits missing ones. `make refit` resamples them. `results/reproduction.json` records execution status.

## What can and cannot be reproduced

Included: exact retained primary annotation inputs, text hashes, frozen scores, bibliography/author metadata, annotation definitions, human-validation records, model code, posterior draws, figures, and numerical outputs.

Not fully recoverable from the release: the original upstream acquisition and random passage-selection process, all provider access dates, raw API responses, retry counts, and complete run provenance for every annotation batch. The exact retained passages can be verified, but reconstructing the original sample from every upstream source is a different task. `analysis/prepare.py` prepares retained French length metadata; it does not resample the original corpus.
