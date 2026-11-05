# Reuse the annotation tools

Reproduction uses frozen scores and requires no API key. Commands in this guide create **new** annotations and may incur provider charges. Model names in the code preserve the study configuration; use `--model` for a model available to your account.

## Input and construct formats

The text tools accept UTF-8 `.txt` files or JSON records with `id` and `content`. JSON may also include `book_id` and `date`:

```json
[{"id": "example-001", "content": "Your passage here.", "date": 1850}]
```

Construct files use one definition per line:

```text
CUSTOM | Example construct | Define the narrative feature to score.
```

Lines beginning with `#` and blank lines are ignored. Use unique construct IDs. The schema assigns each construct an integer `Role` score from 0 to 6.

| Score | Label | Narrative role |
| ---: | :--- | :--- |
| 0 | Absent | Plays no role |
| 1 | Peripheral | No narrative consequence |
| 2 | Background | Context or atmosphere |
| 3 | Contributing | Shapes an event or decision |
| 4 | Significant | Drives a subplot or turning point |
| 5 | Central | Primary narrative driver or motivation |
| 6 | Defining | Dominant theme inseparable from the narrative |

The executed shared prompt is assembled in `src/annotation/annotator.py` from the rubric, instructions, and construct definitions. The files in `prompts/` also preserve reference prompt text. Consult the actual driver when recording a new run's prompt.

## Credentials

Copy `.env.example` to `.env` and fill only the provider key needed, or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in your environment. `.env` and new annotation outputs are ignored by Git. Never put a key in a prompt, example, or issue report.

## Annotate a text directory with GPT

```sh
uv run --locked python script/annotate_dir_gpt.py \
  --input /path/to/texts \
  --ingredients /path/to/constructs.txt \
  --tradition English \
  --model YOUR_MODEL_ID \
  --out results/new_annotations/my_study.json
```

Optional `--metadata` accepts a CSV with `id,date`; IDs must match text filename stems. This driver saves after every input and resumes records containing all requested scores. It retries incomplete records on the next invocation. The custom construct list controls the output fields.

**Use a fresh output path whenever text, prompts, model, or settings change.** Resume checks use IDs and score completeness, not input/configuration hashes. Reusing the same IDs for changed content would otherwise skip that content.

## Annotate JSON batches or use Claude

```sh
uv run --locked python script/run_annotations.py \
  --input /path/to/batch.json \
  --ingredients /path/to/constructs.txt \
  --tradition French \
  --backend openai \
  --model YOUR_MODEL_ID \
  --output-dir results/new_annotations/my_batch
```

Use `--backend claude` with an appropriate model and Anthropic key to switch providers. `--rubric-file` replaces the default rubric; `--seed` sets the best-effort OpenAI seed. This general batch driver currently writes at the end of each batch and does not skip completed records on rerun. Keep batches small enough to restart and use separate output locations.

## Knowledge-based annotation

`script/run_annotations_latent.py` reads bibliographic CSVs and scores works from model knowledge. `script/run_batch_latent.py` submits grouped Claude batch requests. Their `--help` output documents title, author, date, ID, filtering, and output options. These are different protocols from scoring supplied text; do not combine their outputs without retaining that distinction.

The `make annotate-*` targets show the study's corpus-specific commands. They write into `results/new_annotations/` and do not replace frozen analysis inputs. The English Babel batch target is a runnable annotation route, not proof that this exact batch route generated all retained historical outputs.

## Record a new run

Retain the input IDs and hashes, exact prompt and construct file, model/provider, settings, execution date, failures/retries, and output file. Run a small human-checked pilot before processing a new corpus. The existing validation covers short French excerpts and does not automatically validate a new construct or language.
