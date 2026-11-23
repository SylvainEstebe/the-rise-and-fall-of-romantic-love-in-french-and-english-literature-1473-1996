# The rise and fall of romantic love in French and English literature, 1473–1996

Repository for the article by Sylvain Estebe, Edgar Dubourg, Ze Fu and Nicolas Baumard.

[Figures](results/figures/README.md) · [Numerical results](results/README.md) · [Data](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus) · [Prompts](prompts)

![Romantic affiliation](results/figures/PNG_300dpi/Figure_1.png)

## Installation

```sh
uv sync --locked
make spline-figures
```

`make spline-figures` redraws the spline figures from saved tables. `make submission-figures` rebuilds Figures 1 and S1–S7 and requires Poppler (`pdftoppm`). `make check` verifies the sample sizes.

## Data

Texts and scores are in the gated Hugging Face dataset [SylvainEstebe/romantic_love_corpus](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus). Accept its conditions, then set `HF_TOKEN` or run `hf auth login`. The analysis scripts read the scores, authors and dates from it, download only those columns and cache them under `.cache/hf/`.

A local `data/` folder, if present, is used instead; it is not tracked in git. Set `ROMANTIC_LOVE_DATA=hf` or `local` to choose, and `ROMANTIC_LOVE_HF_REVISION` to pin a dataset commit. `make compare-data` checks that both sources give the same tables.

The text hashes in `make check`, the human validation, the French length sensitivity (`make prepare`, `make length`) and new API annotations still need the local `data/` folder.

## Contents

| Content | Location |
| :--- | :--- |
| Main figure and supplementary Figures S1–S7 | [Figure gallery and index](results/figures/README.md) |
| Spline model comparison, curves, diagnostics and timing estimates | [Spline results](results/splines/README.md) |
| Table S10: robustness to time-trend specification | [Readable table](results/splines/robustness/Table_S10.md) · [CSV](results/splines/robustness/Table_S10.csv) |
| Century contrasts and steepest decline | [Contrasts](results/splines/contrasts) |
| Length-adjusted spline sensitivity | [Numerical results](results/splines/robustness/Length_sensitivity.csv) |
| Analysis-ready data | `analysis_data.csv` in each [affiliation](results/splines/affiliation), [English Babel affiliation](results/splines/affiliation_babel) and [signalling](results/splines/signalling) corpus directory |
| Texts and scores | [Hugging Face dataset](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus) |
| Annotation scripts and prompts | [Scripts](script) · [Prompts](prompts) |
| Current analysis and figure code | [analysis/splines](analysis/splines) |
| Earlier quadratic, two-lines, validation and decomposition analyses | [Results guide](results/README.md) |

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

[Code licence](LICENSE) · [Data provenance and rights](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus) · [Citation](CITATION.cff)
