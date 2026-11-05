"""
annotation_claude.py
--------------------
Claude (Anthropic) port of ``LiteraryAnnotator`` for the model benchmark.

It reuses the EXACT same system prompt, ingredient list, rubric/instructions
and 0-6 "Role" schema as the OpenAI annotator (src/annotation/annotator.py),
so GPT and Claude scores are directly comparable. The only difference is the
backend: Anthropic Messages API with a forced tool call for structured output.

Public surface mirrors LiteraryAnnotator.annotate_text():
    ann = ClaudeAnnotator(ingredients, model="claude-opus-4-8")
    scores = ann.annotate_text(text)   # -> {"ING_38": {"Role": 4}, ...} or None
"""

from __future__ import annotations

from anthropic import Anthropic

# Reuse the identical prompt machinery AND the batch/directory orchestration of
# the GPT annotator: ClaudeAnnotator subclasses LiteraryAnnotator and only swaps
# the per-text API call, so it is a drop-in backend everywhere LiteraryAnnotator
# is used (process_batch / process_directory / export_per_ingredient inherited).
from src.annotation.annotator import (
    Ingredient,
    LiteraryAnnotator,
    DEFAULT_RUBRIC,
    DEFAULT_INSTRUCTIONS,
    build_system_prompt,
)

# Friendly aliases -> concrete Anthropic model ids (override via --model).
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

TOOL_NAME = "record_scores"


def build_tool(ingredients: list[Ingredient]) -> dict:
    """A single forced tool whose schema is one 0-6 integer per ingredient."""
    props = {
        ing.id: {
            "type": "integer",
            "minimum": 0,
            "maximum": 6,
            "description": f"0-6 Role score for {ing.id} {ing.name}.",
        }
        for ing in ingredients
    }
    return {
        "name": TOOL_NAME,
        "description": "Record the 0-6 Role score for each narrative ingredient.",
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": [ing.id for ing in ingredients],
        },
    }


class ClaudeAnnotator(LiteraryAnnotator):
    """Claude backend, drop-in for LiteraryAnnotator (same annotate_text output).

    Inherits process_batch / process_directory / export_per_ingredient unchanged;
    only the per-text API call differs (Anthropic Messages + forced tool call).
    """

    def __init__(
        self,
        ingredients: list[Ingredient],
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        temperature: float = 0.0,
        rubric: str = DEFAULT_RUBRIC,
        instructions: str = DEFAULT_INSTRUCTIONS,
        seed: int | None = None,  # accepted for interface parity; unused by Claude
        max_tokens: int = 1024,
    ):
        # NB: deliberately do NOT call super().__init__ (it builds an OpenAI
        # client). We set the same attributes the inherited methods rely on.
        self.ingredients = ingredients
        self.model = MODEL_ALIASES.get(model, model)
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = build_system_prompt(ingredients, rubric, instructions)
        self.tool = build_tool(ingredients)

    def _create(self, text: str, with_temperature: bool):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            tools=[self.tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": text}],
        )
        if with_temperature and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return self.client.messages.create(**kwargs)

    def annotate_text(self, text: str) -> dict | None:
        """Return {ING_id: {"Role": int}} (mirrors LiteraryAnnotator) or None."""
        try:
            resp = self._create(text, with_temperature=True)
        except Exception as e:  # noqa: BLE001
            # Newer models (e.g. claude-opus-4-8) deprecate `temperature`;
            # retry once without it rather than failing the work.
            if "temperature" in str(e).lower():
                try:
                    resp = self._create(text, with_temperature=False)
                except Exception as e2:  # noqa: BLE001
                    print(f"  ⚠ Erreur API Claude ({self.model}) : {e2}")
                    return None
            else:
                print(f"  ⚠ Erreur API Claude ({self.model}) : {e}")
                return None

        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                raw = block.input or {}
                out: dict[str, dict] = {}
                for ing in self.ingredients:
                    val = raw.get(ing.id)
                    out[ing.id] = {"Role": int(val) if val is not None else None}
                return out
        print(f"  ⚠ Pas de tool_use dans la réponse ({self.model}).")
        return None
