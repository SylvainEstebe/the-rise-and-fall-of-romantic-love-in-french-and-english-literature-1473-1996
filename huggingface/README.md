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

# Romantic love in French and English literature, 1473–1996

Texts used in the article "The rise and fall of romantic love in French and English literature, 1473–1996". Annotations, metadata, validation materials and analysis code are in the [GitHub repository](https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996).

## Collections

| Collection | Identifier | Records | Years | Format |
|:---|:---|---:|:---|:---|
| French books (Roland) | `roland_fr` | 1,879 | 1500–1996 | JSON batches; text in the `content` field |
| English books (EarlyPrint–Gutenberg) | `novels_uk` | 1,887 | 1473–1949 | `.txt` files |
| French theatre | `theatre_fr` | 1,060 | 1574–1899 | `.txt` files |
| English theatre | `theatre_uk` | 932 | 1576–1800 | `.txt` files |

5,758 records in total. Book records are sampled passages, not complete books.

## Download

The ZIP archive under [Files and versions](https://huggingface.co/datasets/SylvainEstebe/romantic_love_corpus/tree/main) contains the texts only.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SylvainEstebe/romantic_love_corpus",
    repo_type="dataset",
    allow_patterns=["*.zip"],
    local_dir="romantic_love_corpus",
)
```

If the repository is private, run `hf auth login` first.

## Annotations

The GitHub repository holds scores for the narrative importance of five themes on a 0–6 scale, from absent to defining: romantic affiliation, long-term love, romantic signalling, parental care and parental investment. Dates and authors are in the metadata files of each collection; theatre annotation records have no dates. `data/metadata/text_index.csv` maps record identifiers to files and gives SHA-256 hashes.

## Limitations

- The collections differ in period, genre and source.
- The texts contain OCR errors and historical spelling.
- A sampled passage may not reflect the whole work.
- Scores are model-generated; see the prompts and the human validation.

## Rights

No dataset-wide licence is declared. The MIT licence of the GitHub repository covers the software, not the literary texts. Keep source attribution when reusing texts, and record the dataset revision used.
