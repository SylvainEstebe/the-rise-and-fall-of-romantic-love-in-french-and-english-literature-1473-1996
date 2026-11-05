"""
src.annotation
--------------
Literary annotation package.

Re-exports the public members of the annotator module for shorter imports:

    from src.annotation import LiteraryAnnotator, load_ingredients
"""

from .annotator import (
    Ingredient,
    IngredientScore,
    LiteraryAnnotator,
    build_schema,
    build_system_prompt,
    load_ingredients,
)

__all__ = [
    "Ingredient",
    "IngredientScore",
    "LiteraryAnnotator",
    "build_schema",
    "build_system_prompt",
    "load_ingredients",
]
