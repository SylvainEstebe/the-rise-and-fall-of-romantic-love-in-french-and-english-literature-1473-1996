"""Recover French source-length metadata from the exact annotation batches."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = [
        row
        for path in sorted((ROOT / "data/texts/roland_fr").glob("batch*.json"))
        for row in json.loads(path.read_text())
    ]
    columns = ["id", "book_id", "date", "chapter_idx", "n_chapters", "char_count"]
    metadata = pd.DataFrame(rows)[columns]
    scores = pd.read_json(ROOT / "data/annotations/main/roland_fr/annotations_gpt.json")
    assert metadata.id.is_unique and scores.id.is_unique
    data = scores.merge(metadata, on=["id", "book_id", "date"], validate="one_to_one")
    assert len(data) == 1879
    data["year"] = data.date.astype(int)
    for code in ["ING_38", "ING_40", "TH_03"]:
        data[code] = pd.to_numeric(data[f"{code}_Role"], errors="raise")
    out = ROOT / "data/metadata/roland_length.csv"
    data[
        [
            "id",
            "book_id",
            "year",
            "n_chapters",
            "char_count",
            "chapter_idx",
            "ING_38",
            "ING_40",
            "TH_03",
        ]
    ].to_csv(out, index=False)


if __name__ == "__main__":
    main()
