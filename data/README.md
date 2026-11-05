# Data guide

## Layout

| Path | Contents |
| :--- | :--- |
| `annotations/main/<corpus>/` | Frozen text-based scores and matching bibliographic metadata |
| `annotations/supplementary/babel_fr/` | French bibliographic metadata and GPT knowledge-based ratings |
| `annotations/supplementary/babel_uk/` | English bibliographic metadata and partial Claude knowledge-based ratings |
| `texts/roland_fr/batch*.json` | Exact French passage inputs, in each record's `content` field |
| `texts/novels_uk/`, `texts/theatre_fr/`, `texts/theatre_uk/` | Exact plain-text inputs |
| `metadata/text_index.csv` | Primary input inventory and SHA-256 checksums |
| `metadata/roland_length.csv` | Derived French source-length variables and selected scores |
| `validation/` | Human ratings, GPT ratings, and the 40 retained validation inputs |

Paths in this table are relative to `data/`; paths stored inside `text_index.csv` are relative to the repository root.

## Main corpora

| Identifier | Records | Years | Source and input unit |
| :--- | ---: | :--- | :--- |
| `roland_fr` | 1,879 | 1500–1996 | Roland; one retained sampled passage per source |
| `novels_uk` | 1,887 | 1473–1949 | EarlyPrint–Gutenberg; one retained passage per work |
| `theatre_fr` | 1,060 | 1574–1899 | Théâtre Classique; complete available play texts |
| `theatre_uk` | 932 | 1576–1800 | British theatre collection; complete available play texts |

French passages are approximately 15,000 characters. English passages were selected from pre-segmented parts of at least 300 words and capped at 4,000 words. They are not uniformly 300-word excerpts. A passage-level zero does not establish absence from the complete work.

## Score records

JSON annotation files contain a list of records. Common fields:

| Field | Meaning |
| :--- | :--- |
| `id` | Record identifier; join as a string within the relevant corpus |
| `book_id` | Upstream/source identifier when available; may be null |
| `date` | Annotation-side publication date; may be null, especially for plays |
| `<construct>_Role` | Integer narrative-importance score from 0 to 6; null means missing |
| `title`, `author` | Included directly in Babel outputs; otherwise use metadata |
| `_known` | Babel familiarity flags by construct; false/unknown is not a zero score |
| `_explanations` | Babel explanations when retained; not present for all providers/runs |

Use the matching metadata CSV for dates and authors in the primary analyses. Preserve IDs as strings and check join uniqueness. Use `(corpus, id)` across collections. Missing scores must not be converted to zero for general analysis.

The five shared constructs are `ING_38` (romantic affiliation), `ING_40` (long-term love), `TH_01` (parental care), `TH_02` (parental investment), and `TH_03` (romantic signalling). Some corpora also include `ING_41`, `ING_42`, and `ING_43`. Exact definitions are in [themes.txt](../prompts/themes.txt) and [themes_8.txt](../prompts/themes_8.txt).

## Metadata and hashes

Primary metadata includes `id`, `title`, `author`, and `date`, plus source-specific fields. French book metadata also retains catalogue date conflicts, matching status, and original batch identifiers. Use the analysis date field rather than extracting a year from a filename.

`text_index.csv` contains `corpus`, `id`, `path`, `json_field`, and `sha256`. For French JSON batches, the checksum covers the UTF-8 bytes of the selected record's `content`; for plain-text files, it covers file bytes. `make check` verifies all 5,758 entries.

`roland_length.csv` contains `id`, `book_id`, `year`, `n_chapters`, `char_count`, `chapter_idx`, and selected construct scores. `n_chapters` counts available source segments, not verified complete-book length. `char_count` describes the retained passage. `analysis/prepare.py` regenerates this file from the retained batches.

## Babel coverage

The retained annotation files contain 4,186 French rows and 978 English rows, including missing ratings. The analytical romantic-affiliation samples contain 1,870 and 826 records; romantic-signalling samples contain 1,946 and 799. Filtering depends on valid scores, familiarity, metadata, and the analysis loader. Raw row counts are not analytical sample sizes. Refer to `analysis/latent_corpora.py` for the exact filters.

Babel ratings are based on model knowledge, not supplied texts. French and English runs also use different model providers. They should be kept separate from the four primary text-based datasets.

## Human-validation correction

The validation script documents a historical interface problem that failed to record zero scores for two constructs for the coder stored as `sylvain`. To reproduce the retained results, `analysis/validation.py` restores those blanks to zero for `ING_38` and `ING_40`. For `ING_38`, 19 of the 40 raw ratings are blank. The raw files are preserved; the correction is applied during analysis.

This is a study-specific correction, not a general missing-data rule. The present repository contains the correction in code but not an independent interface event log proving the meaning of each blank. The reported human–human affiliation kappa is approximately 0.744 after correction. Restricting to the 21 complete raw pairs gives approximately 0.688; these are different samples.

## Sources and rights

| Material | Source/provenance |
| :--- | :--- |
| Roland texts | Digitized French literary sources from Gallica, HathiTrust, and Google Books; collection and source metadata accompany the scores |
| English books | [EarlyPrint](https://texts.earlyprint.org/home.html) and [Project Gutenberg](https://www.gutenberg.org/) |
| French theatre | [Théâtre Classique](https://www.theatre-classique.fr/) |
| English theatre | Folger collections and the theatre corpus described by [Martins & Baumard (2020)](https://doi.org/10.1073/pnas.2009571117) |
| Babel | Bibliographic records retained in the supplementary metadata CSVs |
| Two-lines implementation | Archived R implementation from [Simonsohn's two-lines project](https://webstimate.org/twolines/); retained in `analysis/twolines.R` |

The repository's MIT license covers its software; it does not grant blanket rights to literary texts, source editions, or third-party materials. A complete source-by-source redistribution-permission inventory is not included. Preserve attribution and establish the applicable permissions for the particular materials being redistributed. No dataset-wide text license is declared here.

See [reproduction limits](../docs/REPRODUCIBILITY.md#what-can-and-cannot-be-reproduced) for upstream sampling and annotation-run records that were not retained.
