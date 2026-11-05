#!/usr/bin/env python3
"""Resumable GPT annotation of a directory of text files."""

import argparse
import csv
import json
import os
import pathlib
import sys
import time

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.annotation.annotator import DEFAULT_RUBRIC, LiteraryAnnotator, load_ingredients  # noqa: E402

def atomic_save(path: pathlib.Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def annotate_dir(annotator, files, out_file: pathlib.Path, dates: dict):
    constructs = [ingredient.id for ingredient in annotator.ingredients]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_file.read_text()) if out_file.exists() else []
    done = {
        r["id"] for r in results if all(r.get(f"{c}_Role") is not None for c in constructs)
    }
    todo = [f for f in files if f.stem not in done]
    print(
        f"  {out_file.name}: {len(files)} files | {len(done)} done | {len(todo)} to annotate",
        flush=True,
    )
    fails, t0 = 0, time.time()
    for i, f in enumerate(todo):
        ai = annotator.annotate_text(f.read_text(encoding="utf-8", errors="ignore"))
        row = {"id": f.stem, "book_id": None, "date": dates.get(f.stem)}
        for c in constructs:
            row[f"{c}_Role"] = ai[c]["Role"] if ai else None
        fails = fails + 1 if ai is None else 0
        results = [r for r in results if r["id"] != row["id"]] + [row]
        atomic_save(out_file, results)
        rate = (time.time() - t0) / (i + 1)
        print(
            f"    [{i + 1}/{len(todo)}] {f.stem[:45]:45s} ETA {rate * (len(todo) - i - 1) / 3600:.1f}h",
            flush=True,
        )
        if fails >= 5:
            sys.exit(
                "!! 5 échecs consécutifs (clé/API ?) — arrêt ; relance = reprise auto."
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help=".json (flat input) or a directory (input with subdirs)",
    )
    ap.add_argument(
        "--tradition", default="English", help="'English' or 'theatre' etc."
    )
    ap.add_argument("--model", default="gpt-5.4-2026-03-05")
    ap.add_argument(
        "--ingredients", type=pathlib.Path, default=ROOT / "prompts/themes_8.txt"
    )
    ap.add_argument(
        "--metadata", type=pathlib.Path, default=None, help="CSV with id,date"
    )
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY absent de .env")
    dates = {}
    if a.metadata and a.metadata.exists():
        dates = {r["id"]: r.get("date") for r in csv.DictReader(open(a.metadata))}

    annotator = LiteraryAnnotator(
        ingredients=load_ingredients(a.ingredients),
        model=a.model,
        api_key=key,
        temperature=0.0,
        rubric=DEFAULT_RUBRIC.replace(
            "French literary texts", f"{a.tradition} literary texts"
        ),
    )
    subdirs = sorted(p for p in a.input.iterdir() if p.is_dir())
    if subdirs:
        for sd in subdirs:
            annotate_dir(
                annotator, sorted(sd.glob("*.txt")), a.out / f"{sd.name}.json", dates
            )
    else:
        annotate_dir(annotator, sorted(a.input.glob("*.txt")), a.out, dates)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
