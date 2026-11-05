"""Offline checks for configurable annotation and the public example."""

import json
from pathlib import Path
import tempfile
import unittest

from examples.summarize_scores import summarize
from script.annotate_dir_gpt import annotate_dir
from src.annotation.annotator import Ingredient


class FakeAnnotator:
    ingredients = [Ingredient("CUSTOM", "Example", "A custom construct")]

    def __init__(self):
        self.calls = 0

    def annotate_text(self, text):
        self.calls += 1
        return {"CUSTOM": {"Role": 3}}


class ReuseTests(unittest.TestCase):
    def test_custom_construct_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "work.txt"
            source.write_text("An invented example.", encoding="utf-8")
            output = root / "scores.json"
            annotator = FakeAnnotator()
            annotate_dir(annotator, [source], output, {"work": "1800"})
            annotate_dir(annotator, [source], output, {"work": "1800"})
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(json.loads(output.read_text()), [{"id": "work", "book_id": None, "date": "1800", "CUSTOM_Role": 3}])

    def test_missing_scores_are_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scores.json"
            source.write_text(json.dumps([{"CUSTOM_Role": 0}, {"CUSTOM_Role": 4}, {"CUSTOM_Role": None}]))
            result = summarize(source, "CUSTOM")
            self.assertEqual(result["observed"], 2)
            self.assertEqual(result["missing"], 1)
            self.assertEqual(result["mean_score"], 2)
            self.assertEqual(result["proportion_positive"], 0.5)

    def test_unknown_construct_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scores.json"
            source.write_text('[{"CUSTOM_Role": 3}]')
            with self.assertRaisesRegex(ValueError, "No MISSING_Role"):
                summarize(source, "MISSING")


if __name__ == "__main__":
    unittest.main()
