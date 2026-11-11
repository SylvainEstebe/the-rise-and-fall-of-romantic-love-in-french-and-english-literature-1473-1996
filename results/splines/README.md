# Spline results

[Home](../../README.md) · [Reproduce](../../docs/SPLINES.md) · [Submission figures](../figures/README.md)

| Folder | Contents |
| :--- | :--- |
| [affiliation](affiliation) | French/English books and theatre, plus French Babel: linear, quadratic and cubic spline df 3–6 |
| [affiliation_babel/english_babel](affiliation_babel/english_babel) | English Babel affiliation, same six specifications |
| [signalling](signalling) | Six corpora, spline df 4–6 |
| [robustness](robustness) | Table S10, Figure S7, length-adjusted sensitivity and posterior timing summaries |
| [contrasts](contrasts) | Century contrasts, steepest-decline dates and Babel bootstrap results |
| [figure_data](figure_data) | Displayed-model selections, plotted curves, descriptive bins and source manifests |
| [figures](figures) | Individual spline PNG/PDF plots and captions |

Each corpus folder contains `analysis_data.csv`, `loo_full.csv`, `all_curves.csv`, `all_model_quantities.csv`, diagnostics, sampling attempts and provenance. `loo_full.csv` includes exact corrections where applicable; uncorrected PSIS and common-row sensitivity comparisons are retained separately. Full posterior `.nc` files and predictive `.npz` arrays are distributed in release v0.2.0; [restoration instructions](../../docs/SPLINES.md) explain the two archive sets.

[Table S10](robustness/Table_S10.md) · [Full-precision Table S10](robustness/Table_S10_numeric_long.csv) · [Length sensitivity](robustness/Length_sensitivity.csv) · [Century contrasts](contrasts/spline_contrasts.csv) · [Steepest decline](contrasts/steepest_decline.csv)

`import_manifest.json` records the original November output paths and file hashes before/after path relocation. Statistical CSV values and posterior binaries were preserved. Original provenance records describe the original run; portable script locations and entry points are documented above. `binary_manifest.csv` and `binary_archives.json` record the release files and SHA-256 checksums. `reproduction.json` records the latest figure-only verification, not a new MCMC run.
