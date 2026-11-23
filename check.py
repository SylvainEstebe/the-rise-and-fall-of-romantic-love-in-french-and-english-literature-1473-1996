"""Check the retained data, text hashes and manuscript sample sizes offline."""

from pathlib import Path
import ast
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import four_corpora
import latent_corpora
import text_signaling
import validation
import data_source
from src.annotation.annotator import build_schema, build_system_prompt, load_ingredients


def check_local_inputs(counts):
    """Text hashes and human validation, from the local data/ folder only."""
    batches = {}
    for path in (ROOT / "data/texts/roland_fr").glob("batch*.json"):
        for row in json.loads(path.read_text()):
            assert row["id"] not in batches
            batches[row["id"]] = row
    seen = set()
    for row in csv.DictReader((ROOT / "data/metadata/text_index.csv").open()):
        path = ROOT / row["path"]
        assert path.is_file(), path
        content = (
            batches[row["id"]]["content"].encode()
            if row["json_field"]
            else path.read_bytes()
        )
        assert hashlib.sha256(content).hexdigest() == row["sha256"], path
        assert (row["corpus"], row["id"]) not in seen
        seen.add((row["corpus"], row["id"]))
    expected = {
        (name, str(row_id))
        for name in counts
        for row_id in four_corpora.load_corpus(name).id
    }
    assert seen == expected
    human = validation.pair(
        validation.load_human("arman"), validation.load_human("sylvain", True)
    )
    np.testing.assert_allclose(human["kappa_quadratic"], 0.8302350052613119)


def main():
    import pytensor
    import pytensor.tensor as pt

    x = pt.dscalar("compiler_check")
    evaluate = pytensor.function([x], x * x + 1)
    assert evaluate(3.0) == 10.0
    counts = {
        name: len(four_corpora.load_corpus(name)) for name in four_corpora.CORPORA
    }
    assert counts == {
        "roland_fr": 1879,
        "novels_uk": 1887,
        "theatre_fr": 1060,
        "theatre_uk": 932,
    }
    assert {
        name: len(text_signaling.load_corpus(ROOT, name)) for name in counts
    } == counts
    latent = {
        name: len(latent_corpora.load_corpus(name)) for name in latent_corpora.ANALYSES
    }
    assert latent == {
        "babel_fr_ING_38": 1870,
        "babel_uk_ING_38": 826,
        "babel_fr_TH_03": 1946,
        "babel_uk_TH_03": 799,
    }
    local = data_source.source() == "local"
    if local:
        check_local_inputs(counts)
    else:
        print("Skipped text hashes and human validation: they need the local data/ folder.")
    for file in ["themes.txt", "themes_8.txt"]:
        ingredients = load_ingredients(ROOT / "prompts" / file)
        assert set(build_schema(ingredients).model_fields) == {
            i.id for i in ingredients
        }
        assert all(i.id in build_system_prompt(ingredients) for i in ingredients)
    for directory in ["analysis", "script", "src"]:
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
    checked = "exact inputs, validation and prompts" if local else "prompts"
    print(f"OK: {sum(counts.values())} text records, four Babel samples, {checked}.")


if __name__ == "__main__":
    main()
