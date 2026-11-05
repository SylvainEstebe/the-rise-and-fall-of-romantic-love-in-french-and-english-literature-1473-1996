---
pretty_name: Romantic Love in French and English Literature (1473–1996)
language:
  - fr
  - en
size_categories:
  - 1K<n<10K
tags:
  - text
  - literature
  - digital-humanities
  - cultural-evolution
  - historical-corpora
  - romantic-love
  - computational-literary-studies
---

# Romantic Love in Literature

### French & English · 1473–1996

> Five centuries of literary texts for studying how stories represent courtship, devotion, and romantic love.

**5,758 text records · 2 languages · 4 corpora**

[Browse the files](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus/tree/main) · [Companion research repository](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996)

---

## About the corpus

The **Romantic Love Corpus** brings together French and English literary texts used in the research project *The Rise and Fall of Romantic Love in French and English Literature, 1473–1996*.

It supports the study of romantic themes across historical periods, languages, and literary forms. The four main collections cover French books, English books, French theatre, and English theatre.

The corpus is a resource for computational literary studies, digital humanities, and research on cultural change. Texts are included as material for studying romantic themes; inclusion does not imply that every text depicts romantic love.

## At a glance

| Collection | Language | Text records | Publication years |
|:---|:---|---:|:---|
| French books — Roland | French | 1,879 | 1500–1996 |
| English books — EarlyPrint–Gutenberg | English | 1,887 | 1473–1949 |
| French theatre | French | 1,060 | 1574–1899 |
| English theatre | English | 932 | 1576–1800 |
| **Total** | **French & English** | **5,758** | **1473–1996** |

Counts refer to indexed text records in the companion project's four main corpora. They should not be interpreted as counts of complete books: the French book collection contains sampled chapters. The collections cover different historical periods.

## Get the data

**The ZIP archive contains the corpus texts only.** Annotations, bibliographic metadata, validation materials, and analysis code are provided separately in the [companion research repository](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996).

Open **[Files and versions](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus/tree/main)**, download the ZIP archive, and extract it locally.

To download the archive programmatically, install `huggingface_hub` and run:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SylvainEstebe/romantic_love_corpus",
    repo_type="dataset",
    allow_patterns=["*.zip"],
    local_dir="romantic_love_corpus",
)
```

If the repository is private, first sign in with `hf auth login` using an account with access. This downloads the archive; extraction is a separate step.

## Text formats and companion resources

In the companion research repository, the main texts are organized as follows:

| Collection identifier | Format | Text location |
|:---|:---|:---|
| `roland_fr` | JSON batches | The `content` field of each record |
| `novels_uk` | Plain text | Individual `.txt` files |
| `theatre_fr` | Plain text | Individual `.txt` files |
| `theatre_uk` | Plain text | Individual `.txt` files |

The companion repository also provides bibliographic metadata, annotation prompts, model-generated scores, human-validation materials, and analysis scripts. Its `data/metadata/text_index.csv` maps text identifiers to source files and includes SHA-256 hashes.

For analyses involving dates or authors, use the bibliographic metadata associated with each collection. Theatre annotation records have missing dates; dates are supplied in the corresponding metadata files.

## Studying romantic themes

The companion project's core annotation scheme measures the narrative importance of five themes:

| Theme | What it captures |
|:---|:---|
| **Romantic affiliation** | Pursuit of a new romantic connection through courtship or flirtation |
| **Long-term love** | Enduring commitment and devotion beyond initial attraction |
| **Romantic signaling** | Overt, idealizing displays intended to demonstrate romantic commitment |
| **Parental care** | Protection of a child |
| **Parental investment** | Nurturing or raising a child |

Scores use an ordinal **0–6 scale**: absent, peripheral, background, contributing, significant, central, and defining. These scores describe a theme's role in the narrative, rather than word frequency or positive versus negative sentiment. Additional annotation fields are available for some collections; coverage varies.

## Interpretation and limitations

- **Historical coverage is uneven.** The collections differ in period, genre, and source composition. Comparisons should account for these differences.
- **Digitization affects the text.** OCR errors, historical spelling, and editorial conventions may affect automated analysis.
- **Excerpts have limited context.** A sampled chapter may not capture the themes of an entire work.
- **Annotations are estimates.** Model-generated scores should be assessed alongside the prompts and available human-validation materials.
- **Literature is a selective record.** Patterns in these collections do not directly measure historical populations' beliefs or behavior. Historical texts may also contain discriminatory language or representations.

## Rights and attribution

A dataset-wide license is not specified in this card. The companion repository includes an MIT license for its software; this card does not assign that license to the underlying literary texts.

When referring to this resource, credit **Sylvain Estebe**, link to this dataset and the companion research repository, and record the dataset revision used. Retain source attribution when using individual texts.
