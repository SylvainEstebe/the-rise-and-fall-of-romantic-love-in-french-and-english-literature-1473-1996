"""Summarize frozen annotations without an API key, R, or third-party packages."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def summarize(input_path, construct):
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    field = f"{construct}_Role"
    if not rows or not any(field in row for row in rows):
        raise ValueError(f"No {field} field found in {input_path}")
    scores = [row[field] for row in rows if row.get(field) is not None]
    if not scores:
        raise ValueError(f"No observed scores for {construct}")
    if any(type(score) is not int or not 0 <= score <= 6 for score in scores):
        raise ValueError("Scores must be integers between 0 and 6, or null")
    return {
        "construct": construct,
        "records": len(rows),
        "observed": len(scores),
        "missing": len(rows) - len(scores),
        "mean_score": mean(scores),
        "proportion_positive": sum(score > 0 for score in scores) / len(scores),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/annotations/main/roland_fr/annotations_gpt.json")
    parser.add_argument("--construct", default="ING_38")
    parser.add_argument("--output", type=Path, default=ROOT / "results/examples/summary.csv")
    args = parser.parse_args()
    result = summarize(args.input, args.construct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
