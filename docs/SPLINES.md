# Spline analysis

[Home](../README.md) · [Results](../results/splines/README.md) · [Figures](../results/figures/README.md)

Python 3.11. Install with `uv sync --locked`. Run commands from the repository root.

| Task | Command | Additional inputs |
| :--- | :--- | :--- |
| Redraw current spline figures | `make spline-figures` | None; uses committed curves and frozen observations |
| Audit plotted means and credible bands against all 16,000 draws | `make spline-verify` | `script/restore_splines.py --kind draws` |
| Recompute century contrasts, decline rates and Babel bootstrap | `make spline-contrasts` | Predictive draws |
| Regenerate Table S10 and length-sensitivity tables | `make spline-publication` | Full model checkpoints |
| Rebuild eight submission PDFs and 300-dpi PNGs | `make submission-figures` | Poppler's `pdftoppm` on PATH |
| Run regression checks | `make test` | None; no sampling or API calls |

Restore binaries with `uv run --locked python script/restore_splines.py --kind draws`, `--kind models` or `--kind all`. `--check` verifies existing files. `--archives /path/to/downloads` uses downloaded ZIPs. Private downloads require `GH_TOKEN` or `GITHUB_TOKEN`.

## Model fitting

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

Existing checkpoints are reused; a checkout without model files starts new fits. `make reproduce` only redraws figures.

Inputs: `results/courtship/*_data.csv`, `results/signaling/data`, `results/latent/data`, `results/length/linked_data.csv`. Analysed rows: `analysis_data.csv` in each spline corpus directory.

## Specifications

Cumulative-logit models; scores 0–6; author random intercept; publication year centred at the corpus mean and divided by 100. Four chains × 4,000 retained draws. Displayed splines: highest leave-one-out ELPD among df 4–6. Robustness comparisons also include linear, quadratic and df 3 specifications.

Predictions set author deviations to zero. Bands: pointwise 95% equal-tailed credible intervals. Descriptive bins: 25 years in the main figures, 20 years in Figure S7.

Earlier quadratic pipeline: [commands](REPRODUCIBILITY.md).
