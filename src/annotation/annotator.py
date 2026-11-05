"""
annotator.py
------------
Reusable module for annotating French literary texts with an OpenAI model
(ChatGPT) using structured output.

Accepted input structures:
- a directory of subdirectories (each subdirectory = one batch)
- a directory containing .json or .txt files directly (each file = one batch)
- a .json file (list of {"id", "content", "book_id", "date"} objects)
- a .txt file (one document; the filename becomes the id)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, create_model
from openai import OpenAI


# --------------------------------------------------------------------------- #
# Ingredients                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Ingredient:
    id: str
    name: str
    description: str

    def format_line(self) -> str:
        return f"{self.id} {self.name} — {self.description}"


def load_ingredients(path: str | Path) -> list[Ingredient]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier d'ingrédients introuvable : {path}")

    ingredients: list[Ingredient] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                raise ValueError(
                    f"Ligne {line_no} mal formée dans {path} "
                    f"(attendu 'ID | Nom | Description'): {raw!r}"
                )
            ing_id, name = parts[0], parts[1]
            description = " | ".join(parts[2:])
            ingredients.append(Ingredient(ing_id, name, description))

    if not ingredients:
        raise ValueError(f"Aucun ingrédient chargé depuis {path}")
    return ingredients


# --------------------------------------------------------------------------- #
# Dynamic Pydantic schema                                                     #
# --------------------------------------------------------------------------- #


class IngredientScore(BaseModel):
    Role: int = Field(..., ge=0, le=6, description="Score 0–6 (voir rubrique)")


def build_schema(ingredients: list[Ingredient]) -> type[BaseModel]:
    fields = {ing.id: (IngredientScore, ...) for ing in ingredients}
    return create_model("ThemeAnnotation", **fields)


# --------------------------------------------------------------------------- #
# Prompt                                                                      #
# --------------------------------------------------------------------------- #

DEFAULT_RUBRIC = """\
You are a trained annotator scoring French literary texts (15th–20th c.) for narrative ingredients.
For each ingredient, rate its presence using this 0–6 scale:
0 = Absent | Plays no role
1 = Peripheral | No narrative consequence; the story would be unchanged without it
2 = Background | Provides context or atmosphere; does not cause or resolve any event
3 = Contributing | Causes or shapes at least one plot event or character decision
4 = Significant | Drives a meaningful subplot or a key turning point in the main plot
5 = Central | One of the primary drivers of the main plot; a major character motivation
6 = Defining | The story's dominant theme; inseparable from its core meaning
"""

DEFAULT_INSTRUCTIONS = """\
Some texts may be digitized from manuscripts or early printed books — annotate based on
your best interpretation of the overall meaning. Mentally correct OCR if needed and focus
on narrative content, not surface form.

Return for EACH ingredient ID listed above a single integer "Role" between 0 and 6.
"""


def build_system_prompt(
    ingredients: list[Ingredient],
    rubric: str = DEFAULT_RUBRIC,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> str:
    lines = [rubric.strip(), "", "Ingredients to score:"]
    for ing in ingredients:
        lines.append(f"- {ing.format_line()}")
    lines.append("")
    lines.append(instructions.strip())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Annotator                                                                   #
# --------------------------------------------------------------------------- #

DEFAULT_EXTENSIONS = (".json", ".txt")


class LiteraryAnnotator:
    def __init__(
        self,
        ingredients: list[Ingredient],
        model: str = "gpt-5.4-2026-03-05",
        api_key: str | None = None,
        temperature: float = 0.0,
        rubric: str = DEFAULT_RUBRIC,
        instructions: str = DEFAULT_INSTRUCTIONS,
        seed: int | None = None,
        base_url: str | None = None,
    ):
        self.ingredients = ingredients
        self.model = model
        self.temperature = temperature
        self.seed = seed
        # base_url lets the OpenAI client target any OpenAI-compatible endpoint
        # (DeepSeek: https://api.deepseek.com ; local vLLM/Ollama: http://localhost:...).
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )
        self.schema = build_schema(ingredients)
        self.system_prompt = build_system_prompt(ingredients, rubric, instructions)

    # ================================================================== #
    # API call                                                           #
    # ================================================================== #
    def annotate_text(self, text: str) -> dict | None:
        try:
            kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                "response_format": self.schema,
            }
            # OpenAI supports a best-effort `seed` for reproducibility.
            if self.seed is not None:
                kwargs["seed"] = self.seed
            response = self.client.beta.chat.completions.parse(**kwargs)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                refusal = response.choices[0].message.refusal
                if refusal:
                    print(f"  ⚠ Refus du modèle : {refusal}")
                return None
            return parsed.model_dump()
        except Exception as e:
            print(f"  ⚠ Erreur API : {e}")
            return None

    # ================================================================== #
    # Loading helpers                                                    #
    # ================================================================== #
    @staticmethod
    def _items_from_file(path: Path) -> list[dict]:
        """Return a list of {id, content, ...} items from a file."""
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        if suffix == ".txt":
            return [{"id": path.stem, "content": path.read_text(encoding="utf-8")}]
        raise ValueError(f"Type non supporté : {path.suffix}")

    @classmethod
    def _items_from_dir(
        cls,
        directory: Path,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ) -> list[dict]:
        """Concatenate items from all compatible files in a directory."""
        files = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        items: list[dict] = []
        for f in files:
            items.extend(cls._items_from_file(f))
        return items

    # ================================================================== #
    # Core processing                                                    #
    # ================================================================== #
    def _annotate_items(self, items: list[dict], output_path: Path) -> list[dict]:
        results = []
        n = len(items)
        for i, item in enumerate(items):
            item_id = item.get("id", f"unknown_{i}")
            print(f"  [{i + 1}/{n}] {item_id}")
            ai = self.annotate_text(item.get("content", ""))

            row = {
                "id": item_id,
                "book_id": item.get("book_id"),
                "date": item.get("date"),
            }
            for ing in self.ingredients:
                row[f"{ing.id}_Role"] = ai[ing.id]["Role"] if ai else None
            results.append(row)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {n} texte(s) → {output_path}\n")
        return results

    # ================================================================== #
    # Public API                                                         #
    # ================================================================== #
    def process_batch(
        self,
        input_path: str | Path,
        output_path: str | Path,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ) -> list[dict]:
        """Process one batch: a single file (.json/.txt) OR a directory of files."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        if input_path.is_dir():
            items = self._items_from_dir(input_path, extensions)
        else:
            items = self._items_from_file(input_path)

        if not items:
            print(f"  ⚠ Aucun item dans {input_path}")
            return []
        return self._annotate_items(items, output_path)

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        per_ingredient_dir: str | Path | None = None,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ) -> list[dict]:
        """
        Process a root directory. Structure is auto-detected:
        - subdirectories present -> each subdirectory = one batch
        - otherwise -> each compatible file = one batch
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
        if subdirs:
            batches: list[Path] = subdirs
            mode = "sous-dossiers"
        else:
            batches = sorted(
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            )
            mode = "fichiers"

        if not batches:
            print(f"⚠ Aucun batch trouvé dans {input_dir}")
            return []

        print(f"Mode détecté : {mode} ({len(batches)} batch(es))\n")

        all_results: list[dict] = []
        for batch in batches:
            print(f"=== {batch.name} ===")
            # batch.stem works for files (extension stripped) AND directories (= name)
            output_path = output_dir / f"{batch.stem}.json"
            results = self.process_batch(batch, output_path, extensions)
            all_results.extend(results)

        if per_ingredient_dir:
            self.export_per_ingredient(all_results, per_ingredient_dir)

        print("Terminé.")
        return all_results

    # ================================================================== #
    # Per-ingredient export                                              #
    # ================================================================== #
    def export_per_ingredient(
        self,
        results: list[dict],
        per_ingredient_dir: str | Path,
    ) -> None:
        per_ingredient_dir = Path(per_ingredient_dir)
        per_ingredient_dir.mkdir(parents=True, exist_ok=True)

        for ing in self.ingredients:
            key = f"{ing.id}_Role"
            slim = [
                {
                    "id": r["id"],
                    "book_id": r.get("book_id"),
                    "date": r.get("date"),
                    "Role": r.get(key),
                }
                for r in results
            ]
            out = per_ingredient_dir / f"{ing.id}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(slim, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {ing.id} ({ing.name}) → {out}")
