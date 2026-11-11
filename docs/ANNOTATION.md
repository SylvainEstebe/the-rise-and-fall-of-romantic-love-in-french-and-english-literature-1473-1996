# Annotation

Credentials: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, in the environment or `.env`. Model selection: `--model`. Outputs: `results/new_annotations/`.

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

## Text annotation

```sh
uv run --locked python script/annotate_dir_gpt.py \
  --input /path/to/texts \
  --ingredients /path/to/constructs.txt \
  --tradition English \
  --model YOUR_MODEL_ID \
  --out results/new_annotations/my_study.json
```

Metadata CSV: `id,date`. IDs match text filename stems. The directory driver resumes completed IDs; use a new output path when inputs, prompts or settings change.

```sh
uv run --locked python script/run_annotations.py \
  --input /path/to/batch.json \
  --ingredients /path/to/constructs.txt \
  --tradition French \
  --backend openai \
  --model YOUR_MODEL_ID \
  --output-dir results/new_annotations/my_batch
```

Claude: `--backend claude`. The batch driver saves at the end of a batch and does not skip completed records on rerun.

## Latent annotation

| Driver | Function |
| :--- | :--- |
| `script/run_annotations_latent.py` | Knowledge-based annotation from bibliographic CSVs |
| `script/run_batch_latent.py` | Claude batch requests |
| `script/annotator_latent.py` | Prompt construction and shared annotation code |

Reference prompts: [French](../prompts/prompt_latent_fr.txt), [English](../prompts/prompt_latent_uk.txt). Driver options: `--help`. Corpus commands: `make annotate-*`.

For new runs, retain inputs, prompts, model/provider, settings and outputs. Frozen scores in `data/annotations/` are the inputs to the published analyses.
