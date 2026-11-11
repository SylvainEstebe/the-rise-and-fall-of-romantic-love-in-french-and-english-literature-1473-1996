# Reproducing the spline analysis

[Home](../README.md) · [Results](../results/splines/README.md) · [Figures](../results/figures/README.md)

All commands run from the repository root. `uv sync --locked` installs Python 3.11 dependencies from `uv.lock`.

| Task | Command | Additional inputs |
| :--- | :--- | :--- |
| Redraw current spline figures | `make spline-figures` | None; uses committed curves and frozen observations |
| Audit plotted means and credible bands against all 16,000 draws | `make spline-verify` | `script/restore_splines.py --kind draws` |
| Recompute century contrasts, decline rates and Babel bootstrap | `make spline-contrasts` | Predictive draws |
| Regenerate Table S10 and length-sensitivity tables | `make spline-publication` | Full model checkpoints |
| Rebuild eight submission PDFs and 300-dpi PNGs | `make submission-figures` | Poppler's `pdftoppm` on PATH |
| Run regression checks | `make test` | None; no sampling or API calls |

Prefix restoration commands with `uv run --locked python`. `--kind all` restores both archive sets; `--check` verifies existing files without downloading. Already-downloaded ZIPs can be supplied using `--archives /path/to/downloads`.

## Explicit fitting commands

These can take substantial time. They are separate from `make reproduce` and the figure commands. Existing checkpoints are reused after provenance checks; to fit independently, use a fresh checkout without restored model files. Posterior samples may differ across hardware despite the recorded seed.

```sh
# Affiliation: five original robustness corpora, six trend specifications each
uv run --locked python analysis/splines/ordinal_shape_robustness.py

# Affiliation: English Babel extension
uv run --locked python analysis/splines/fit_english_babel.py

# Signalling: six corpora, spline df 4–6
uv run --locked python analysis/splines/fit_signalling_splines.py

# French-book length-adjusted quadratic and spline df 4–6
uv run --locked python analysis/splines/publication_shape_supplement.py --stage length

# Rebuild publication tables and figures from completed fits
make spline-publication
make spline-figures
make spline-contrasts
```

The frozen inputs used by the fitting scripts are in `results/courtship/*_data.csv`, `results/signaling/data`, `results/latent/data` and `results/length/linked_data.csv`. The resulting `analysis_data.csv` in each spline corpus directory records the actual analysed rows. Original annotations and source texts are in `data/`; the earlier preparation scripts remain in `analysis/`.

## Model and figure conventions

Cumulative-logit models use scores 0–6, an author random intercept and publication year centred at the corpus mean and divided by 100. Cubic B-spline models have df = 3–6; the main displayed curves select only among df = 4–6. Each retained fit has four chains of 4,000 draws. The scripts retain diagnostic checks, retry settings and exact leave-one-out corrections for flagged observations. Manifests record settings and versions.

Predictions set author deviations to zero. Figure bands are pointwise 95% equal-tailed credible intervals conditional on the selected specification. Main figure summaries use 25-year bins; robustness Figure S7 uses 20-year bins. Saved full-precision tables are the source of the default redraw. `--verify-draws` additionally checks those tables against the predictive samples.

## Earlier pipeline

`make legacy-reproduce` or `uv run --locked python reproduce.py --legacy` runs the earlier quadratic/two-lines/decomposition pipeline. It requires R and may fit missing models. `make refit` explicitly resamples that earlier pipeline. These commands do not replace the spline workflow above.

[Historical reproducibility details](REPRODUCIBILITY.md) describe the v0.1.0 models and original preparation/validation steps.

The repository is currently private. Automatic archive downloads require a `GH_TOKEN` or `GITHUB_TOKEN` environment variable with read access. Alternatively, download the ZIPs from the release while signed in and pass their folder with `--archives`; no token is then needed by the restoration script.
