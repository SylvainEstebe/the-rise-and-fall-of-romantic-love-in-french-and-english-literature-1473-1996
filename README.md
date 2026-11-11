# The rise and fall of romantic love in French and English literature, 1473–1996

Sylvain Estebe, Edgar Dubourg, Ze Fu and Nicolas Baumard.

[Figures](results/figures/README.md) · [Numerical results](results/README.md) · [Spline reproduction](docs/SPLINES.md) · [Data](data/README.md) · [Annotation and prompts](docs/ANNOTATION.md)

![Romantic affiliation](results/figures/PNG_300dpi/Figure_1.png)

## Installation

```sh
uv sync --locked
make spline-figures
```

`make spline-figures` redraws the spline figures from saved tables. `make submission-figures` rebuilds Figures 1 and S1–S7 and requires Poppler (`pdftoppm`). `make test` runs the tests.

## Contents

| Content | Location |
| :--- | :--- |
| Main figure and supplementary Figures S1–S7 | [Figure gallery and index](results/figures/README.md) |
| Spline model comparison, curves, diagnostics and timing estimates | [Spline results](results/splines/README.md) |
| Table S10: robustness to time-trend specification | [Readable table](results/splines/robustness/Table_S10.md) · [CSV](results/splines/robustness/Table_S10.csv) |
| Century contrasts and steepest decline | [Contrasts](results/splines/contrasts) |
| Length-adjusted spline sensitivity | [Numerical results](results/splines/robustness/Length_sensitivity.csv) |
| Analysis-ready data | `analysis_data.csv` in each [affiliation](results/splines/affiliation), [English Babel affiliation](results/splines/affiliation_babel) and [signalling](results/splines/signalling) corpus directory |
| Raw texts, annotations, manual validation and metadata | [Data guide](data/README.md) |
| Annotation scripts and prompts | [Annotation guide](docs/ANNOTATION.md) · [Prompts](prompts) |
| Current analysis and figure code | [analysis/splines](analysis/splines) |
| Earlier quadratic, two-lines, validation and decomposition analyses | [Results guide](results/README.md) · [Historical reproduction](docs/REPRODUCIBILITY.md) |

## Posterior draws

[Release v0.2.0](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996/releases/tag/v0.2.0): predictive draws (2.45 GB) and fitted models (14.7 GB).

```sh
uv run --locked python script/restore_splines.py --kind draws
make spline-verify
make spline-contrasts

uv run --locked python script/restore_splines.py --kind models
make spline-publication
```

For private-repository downloads, set `GH_TOKEN` or `GITHUB_TOKEN`, or download the ZIPs through GitHub and use `--archives /path/to/downloads`. The restoration script checks SHA-256 hashes.

Earlier quadratic models: [v0.1.0](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996/releases/tag/v0.1.0).

## Corpora

| Corpus | Romantic affiliation observations |
| :--- | ---: |
| French books (Roland) | 1,879 |
| English books (EarlyPrint–Gutenberg) | 1,887 |
| French theatre | 1,060 |
| English theatre | 932 |
| French Babel | 1,870 |
| English Babel | 826 |

Babel: knowledge-based ratings. Books and theatre: text annotations.

[Code licence](LICENSE) · [Data provenance and rights](data/README.md) · [Citation](CITATION.cff)
