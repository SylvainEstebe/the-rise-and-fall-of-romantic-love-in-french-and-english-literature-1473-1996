"""annotator_latent_claude.py"""

from __future__ import annotations

from typing import Optional

from anthropic import Anthropic

from annotator_latent import LatentLiteraryAnnotator, Ingredient, build_user_message
from src.annotation.annotation_claude import MODEL_ALIASES

TOOL_NAME = "record_score"
TOOL_NAME_GROUPED = "record_scores"

_SCORE_PROP = {
    "score": {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 6,
        "description": "Integer 0-6, or null if the work is unknown to you.",
    },
    "known": {
        "type": "boolean",
        "description": "True if you are familiar with this work, else false.",
    },
}
_EXPL_PROP = {
    "explanation": {
        "type": "string",
        "description": "Brief 2-4 sentence justification of the score.",
    }
}


def build_tool(justify: bool = True) -> dict:
    """Single-ingredient forced tool."""
    props = {**_EXPL_PROP, **_SCORE_PROP} if justify else dict(_SCORE_PROP)
    return {
        "name": TOOL_NAME,
        "description": "Record the 0-6 Role score for the work on this ingredient.",
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
        },
    }


def build_grouped_tool(ingredients: list[Ingredient]) -> dict:
    """One tool that records a score + known flag for EVERY ingredient at once."""
    props = {}
    for ing in ingredients:
        props[f"{ing.id}_score"] = {
            "type": ["integer", "null"],
            "minimum": 0,
            "maximum": 6,
            "description": f"0-6 score for '{ing.name}', or null if the work is unknown.",
        }
        props[f"{ing.id}_known"] = {
            "type": "boolean",
            "description": f"True if you actually know this work well enough to score '{ing.name}'.",
        }
    return {
        "name": TOOL_NAME_GROUPED,
        "description": "Record the 0-6 Role score for the work on EVERY ingredient.",
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
        },
    }


class ClaudeLatentAnnotator(LatentLiteraryAnnotator):
    """Latent annotator using a Claude model instead of OpenAI."""

    def __init__(
        self,
        ingredients,
        model="claude-opus-4-8",
        api_key=None,
        temperature=0.0,
        rubric=None,
        instructions=None,
        seed=None,
        creator_prefix="by",
        date_verb="written in",
        justify=True,
        max_tokens=512,
        group: bool = False,
    ):
        from annotator_latent import (
            DEFAULT_RUBRIC,
            DEFAULT_INSTRUCTIONS,
            build_single_ingredient_prompt,
            build_grouped_prompt,
        )

        rubric = rubric or DEFAULT_RUBRIC
        instructions = instructions or DEFAULT_INSTRUCTIONS
        if not justify:
            instructions = instructions.replace(
                "Provide a brief explanation (2–4 sentences), and return your "
                "judgment in the structured format.",
                "Return your judgment in the structured format.",
            )
        self.ingredients = ingredients
        self.model = MODEL_ALIASES.get(model, model)
        self.temperature = temperature
        self.seed = seed
        self.creator_prefix = creator_prefix
        self.date_verb = date_verb
        self.justify = justify
        self.group = group
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key)
        if group:
            self.tool = build_grouped_tool(ingredients)
            self.grouped_system = build_grouped_prompt(
                ingredients, rubric, instructions
            )
        else:
            self.tool = build_tool(justify)
            self.prompts = {
                ing.id: build_single_ingredient_prompt(ing, rubric, instructions)
                for ing in ingredients
            }

    # ---- API call (shared) ----
    def _create(self, system: str, user_msg: str, with_temperature: bool):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[self.tool],
            tool_choice={"type": "tool", "name": self.tool["name"]},
            messages=[{"role": "user", "content": user_msg}],
        )
        if with_temperature and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return self.client.messages.create(**kwargs)

    def _call(self, system: str, user_msg: str, ctx: str) -> Optional[dict]:
        """One forced-tool call; returns the tool input dict or None. Drops
        `temperature` automatically for models that reject it (Opus 4.7/4.8)."""
        try:
            resp = self._create(system, user_msg, with_temperature=True)
        except Exception as e:  # noqa: BLE001
            if "temperature" in str(e).lower():
                try:
                    resp = self._create(system, user_msg, with_temperature=False)
                except Exception as e2:  # noqa: BLE001
                    print(f"    ⚠ Erreur API Claude ({ctx}) : {e2}")
                    return None
            else:
                print(f"    ⚠ Erreur API Claude ({ctx}) : {e}")
                return None
        for block in resp.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and block.name == self.tool["name"]
            ):
                return dict(block.input or {})
        print(f"    ⚠ Pas de tool_use ({ctx}).")
        return None

    # ---- ungrouped: one call per ingredient (inherited annotate_work uses this) ----
    def annotate_one(
        self, ingredient: Ingredient, title, author, date
    ) -> Optional[dict]:
        user = build_user_message(
            title, author, date, self.creator_prefix, self.date_verb
        )
        return self._call(self.prompts[ingredient.id], user, ingredient.id)

    # ---- grouped: ONE call per work, all ingredients ----
    def annotate_work(
        self, work_id, title, author, date, index=None, total=None
    ) -> dict:
        if not self.group:
            return super().annotate_work(work_id, title, author, date, index, total)
        hdr = f"[{index}/{total}] {work_id}" if index and total else work_id
        print(f"  {hdr} — '{title}' by {author} ({date})  [grouped]")
        user = build_user_message(
            title, author, date, self.creator_prefix, self.date_verb
        )
        parsed = self._call(self.grouped_system, user, "grouped")
        row = {
            "id": work_id,
            "book_id": work_id,
            "date": date,
            "title": title,
            "author": author,
            "_explanations": {},
            "_known": {},
        }
        for ing in self.ingredients:
            if parsed is None:
                row[f"{ing.id}_Role"] = None
                row["_known"][ing.id] = None
            else:
                known = parsed.get(f"{ing.id}_known")
                row[f"{ing.id}_Role"] = parsed.get(f"{ing.id}_score") if known else None
                row["_known"][ing.id] = known
            row["_explanations"][ing.id] = None
        return row
