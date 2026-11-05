# Offline example

Run `python3 examples/summarize_scores.py` from the repository root. No third-party packages or API keys are needed. The script reads frozen French romantic-affiliation scores and writes an ignored CSV under `results/examples/`.

To reuse it with another corpus or construct:

```sh
python3 examples/summarize_scores.py \
  --input data/annotations/main/novels_uk/annotations_gpt.json \
  --construct TH_03 \
  --output results/examples/english_signalling.csv
```

The output reports record count, observed/missing counts, mean score, and positive-score proportion. Missing scores are excluded, not interpreted as zero. These are raw descriptive summaries; Babel analytical samples apply additional filters, and historical trend models account for more structure.

For generating new scores, use the [annotation guide](../docs/ANNOTATION.md).
