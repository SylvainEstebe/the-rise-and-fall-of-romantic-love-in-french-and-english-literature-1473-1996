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

Included: retained annotation inputs, text hashes, frozen scores, bibliographic and author metadata, annotation definitions, human-validation records, model code, posterior draws, figures and numerical outputs.

Not retained: the upstream acquisition and random passage selection, provider access dates, raw API responses, retry counts and full run records for each annotation batch. The retained passages can be checked against their hashes, but the original sample cannot be redrawn from the upstream sources. `analysis/prepare.py` prepares the French length metadata from the retained batches; it does not resample the corpus.
