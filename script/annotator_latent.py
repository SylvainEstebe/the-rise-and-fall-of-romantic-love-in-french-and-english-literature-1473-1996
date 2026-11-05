"""annotator_latent.py"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from openai import OpenAI

# Reuses annotator.py's Ingredient definition for consistency.
from src.annotation import Ingredient


# --------------------------------------------------------------------------- #
# Pydantic schema                                                             #
# --------------------------------------------------------------------------- #


class SingleIngredientScore(BaseModel):
    explanation: str = Field(
        ...,
        description="Brief 2–4 sentence justification of the score.",
    )
    score: int | None = Field(
        ...,
        ge=0,
        le=6,
        description="Integer score 0–6, or null if the work is unknown to you.",
    )
    known: bool = Field(
        ...,
        description="True if you are familiar with this work, false otherwise.",
    )


class SingleIngredientScoreNoExpl(BaseModel):
    """Score-only variant (no justification) — ~5x cheaper on output tokens."""

    score: int | None = Field(
        ...,
        ge=0,
        le=6,
        description="Integer score 0–6, or null if the work is unknown to you.",
    )
    known: bool = Field(
        ...,
        description="True if you are familiar with this work, false otherwise.",
    )


# --------------------------------------------------------------------------- #
# Prompt                                                                      #
# --------------------------------------------------------------------------- #

RUBRIC_TEMPLATE = """\
You are a trained annotator scoring {tradition} {work_noun} ({period}) for narrative ingredients.
Rate the presence of the ingredient in the work using this 0–6 scale:
0 = Absent | Plays no role
1 = Peripheral | No narrative consequence; the story would be unchanged without it
2 = Background | Provides context or atmosphere; does not cause or resolve any event
3 = Contributing | Causes or shapes at least one plot event or character decision
4 = Significant | Drives a meaningful subplot or a key turning point in the main plot
5 = Central | One of the primary drivers of the main plot; a major character motivation
6 = Defining | The story's dominant theme; inseparable from its core meaning
"""


def build_rubric(
    tradition: str = "French",
    work_noun: str = "literary works",
    period: str = "9th–21st c.",
) -> str:
    """Build the scoring rubric for a given tradition / medium.

    ``tradition`` is an adjective ("Indian", "Japanese", "French") so the model
    is not told a non-French work is French. ``work_noun`` / ``period`` adapt to
    the medium: books -> "literary works" / "9th–21st c."; films -> "films" /
    "20th–21st c.".
    """
    return RUBRIC_TEMPLATE.format(
        tradition=tradition, work_noun=work_noun, period=period
    )


# Backward-compatible default (French corpus).
DEFAULT_RUBRIC = build_rubric("French")

INSTRUCTIONS_TEMPLATE = """\
You will be given only the title, {creator}, and {date_word} of the {work}.
Score it from your latent knowledge of the {work}'s content, judging the ingredient by the
standards and beliefs of the {work}'s historical period (not modern standards).

If you are NOT familiar with this specific {work}, set known=false and score=null.
Do NOT guess from the title alone — only score {works} whose plot, characters, and themes
you actually know. It is acceptable and expected that many {works} will be unknown.

Provide a brief explanation (2–4 sentences), and return your judgment in the structured format.
"""


def build_instructions(
    creator: str = "author",
    date_word: str = "date",
    work: str = "work",
    works: str = "works",
) -> str:
    """Build the medium-specific instruction block.

    books -> "title, author, and date of the work"; films ->
    "title, director, and release date of the film". Keeps the latent
    title+creator method of Dubourg et al. (2024) identical otherwise.
    """
    return INSTRUCTIONS_TEMPLATE.format(
        creator=creator, date_word=date_word, work=work, works=works
    )


# Backward-compatible default (book corpus).
DEFAULT_INSTRUCTIONS = build_instructions()


def build_single_ingredient_prompt(
    ingredient: Ingredient,
    rubric: str = DEFAULT_RUBRIC,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> str:
    """Build the system prompt for a single-ingredient query."""
    lines = [
        rubric.strip(),
        "",
        "Ingredient to score:",
        f"- {ingredient.format_line()}",
        "",
        instructions.strip(),
    ]
    return "\n".join(lines)


def build_grouped_prompt(
    ingredients: list,
    rubric: str = DEFAULT_RUBRIC,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> str:
    """System prompt listing ALL ingredients, for one grouped call per work
    (rubric sent once instead of once per ingredient)."""
    lines = [
        rubric.strip(),
        "",
        "Score EACH of the following ingredients independently:",
    ]
    lines += [f"- {ing.format_line()}" for ing in ingredients]
    lines += ["", instructions.strip()]
    return "\n".join(lines)


def build_user_message(
    title: str,
    author: str,
    date: str | int | None,
    creator_prefix: str = "by",
    date_verb: str = "written in",
) -> str:
    """The user message simply identifies the work.

    ``creator_prefix`` / ``date_verb`` adapt the wording to the medium:
    books use "by <author>, written in <date>"; films use
    "directed by <director>, released in <date>".
    """
    date_str = str(date) if date else "unknown date"
    return f"The work is: '{title}' {creator_prefix} {author}, {date_verb} {date_str}."


# --------------------------------------------------------------------------- #
# Annotator                                                                   #
# --------------------------------------------------------------------------- #


class LatentLiteraryAnnotator:
    """Annotate works by title+author, one ingredient per API call."""

    def __init__(
        self,
        ingredients: list[Ingredient],
        model: str = "gpt-4o-2024-08-06",
        api_key: str | None = None,
        temperature: float = 0.0,
        rubric: str = DEFAULT_RUBRIC,
        instructions: str = DEFAULT_INSTRUCTIONS,
        seed: int | None = None,
        creator_prefix: str = "by",
        date_verb: str = "written in",
        justify: bool = True,
    ):
        self.ingredients = ingredients
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.creator_prefix = creator_prefix
        self.date_verb = date_verb
        self.justify = justify
        # Drop the justification (cheaper) when justify=False.
        self.response_format = (
            SingleIngredientScore if justify else SingleIngredientScoreNoExpl
        )
        if not justify:
            instructions = instructions.replace(
                "Provide a brief explanation (2–4 sentences), and return your "
                "judgment in the structured format.",
                "Return your judgment in the structured format.",
            )
        self.client = OpenAI(api_key=api_key)
        self.prompts = {
            ing.id: build_single_ingredient_prompt(ing, rubric, instructions)
            for ing in ingredients
        }

    # ================================================================== #
    # API call: one request for ONE ingredient                           #
    # ================================================================== #
    def annotate_one(
        self,
        ingredient: Ingredient,
        title: str,
        author: str,
        date: str | int | None,
    ) -> Optional[dict]:
        """Score one work on one ingredient. Returns dict or None on error."""
        user_msg = build_user_message(
            title, author, date, self.creator_prefix, self.date_verb
        )
        try:
            kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": self.prompts[ingredient.id]},
                    {"role": "user", "content": user_msg},
                ],
                "response_format": self.response_format,
            }
            if self.seed is not None:
                kwargs["seed"] = self.seed
            response = self.client.beta.chat.completions.parse(**kwargs)
            parsed = response.choices[0].message.parsed
            if parsed is None:
                refusal = response.choices[0].message.refusal
                if refusal:
                    print(f"    ⚠ Refus du modèle : {refusal}")
                return None
            return parsed.model_dump()
        except Exception as e:
            print(f"    ⚠ Erreur API ({ingredient.id}) : {e}")
            return None

    # ================================================================== #
    # Annotate one work on all ingredients                               #
    # ================================================================== #
    def annotate_work(
        self,
        work_id: str,
        title: str,
        author: str,
        date: str | int | None,
        index: int | None = None,
        total: int | None = None,
    ) -> dict:
        """
        Score one work on all ingredients (= len(ingredients) API calls).
        Returns a row compatible with annotator.py's output format.
        """
        header = (
            f"[{index}/{total}] {work_id}"
            if index is not None and total is not None
            else work_id
        )
        print(f"  {header} — '{title}' by {author} ({date})")

        row = {
            "id": work_id,
            "book_id": work_id,
            "date": date,
            "title": title,
            "author": author,
        }
        row["_explanations"] = {}
        row["_known"] = {}

        for ing in self.ingredients:
            result = self.annotate_one(ing, title, author, date)
            if result is None:
                row[f"{ing.id}_Role"] = None
                row["_explanations"][ing.id] = None
                row["_known"][ing.id] = None
            else:
                if not result.get("known", True):
                    row[f"{ing.id}_Role"] = None
                else:
                    row[f"{ing.id}_Role"] = result.get("score")
                row["_explanations"][ing.id] = result.get("explanation")
                row["_known"][ing.id] = result.get("known")

        return row

    # ================================================================== #
    # Loading the matching CSV                                           #
    # ================================================================== #
    @staticmethod
    def load_works_csv(
        csv_path: str | Path,
        id_col: str = "roland_id",
        title_col: str = "babel_title",
        author_col: str = "babel_author",
        date_col: str = "babel_date",
        min_date: int | None = None,
        max_date: int | None = None,
        country_col: str | None = None,
        country: str | None = None,
    ) -> list[dict]:
        """
        Load a CSV and return a list of work descriptors.
        Works for babel_roland_matches_strict.csv (default columns) and for
        babel_french.csv (pass --id-col title_id etc.).

        If min_date / max_date are given, works outside [min_date, max_date]
        are excluded. Works with an unparseable date are also excluded when
        any date bound is set (we cannot place them on the timeline).

        If ``country`` and ``country_col`` are both given, only rows whose
        ``country_col`` equals ``country`` (case-sensitive, trimmed) are kept —
        e.g. annotate only the Indian or Japanese subset of a babel CSV.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")

        date_filter = (min_date is not None) or (max_date is not None)
        country_filter = bool(country_col) and bool(country)

        works = []
        seen = set()
        n_excluded_date = 0
        n_excluded_unparseable = 0
        n_excluded_country = 0
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if country_filter and country_col not in (reader.fieldnames or []):
                raise KeyError(
                    f"Colonne pays '{country_col}' absente du CSV. "
                    f"Colonnes disponibles : {reader.fieldnames}"
                )
            for row in reader:
                work_id = (row.get(id_col) or "").strip()
                if not work_id or work_id in seen:
                    continue

                # Country filtering
                if country_filter and (row.get(country_col) or "").strip() != country:
                    n_excluded_country += 1
                    continue

                date_raw = (row.get(date_col) or "").strip()

                # Date filtering
                if date_filter:
                    try:
                        year = int(float(date_raw))
                    except (TypeError, ValueError):
                        n_excluded_unparseable += 1
                        continue
                    if min_date is not None and year < min_date:
                        n_excluded_date += 1
                        continue
                    if max_date is not None and year > max_date:
                        n_excluded_date += 1
                        continue

                seen.add(work_id)
                works.append(
                    {
                        "id": work_id,
                        "title": (row.get(title_col) or "").strip(),
                        "author": (row.get(author_col) or "").strip(),
                        "date": date_raw,
                    }
                )

        if country_filter:
            print(
                f"  Filtre pays '{country}' (colonne {country_col}) : "
                f"{n_excluded_country} oeuvre(s) d'un autre pays -- exclues."
            )
        if date_filter:
            bounds = (
                f"[{min_date if min_date is not None else '-inf'}, "
                f"{max_date if max_date is not None else '+inf'}]"
            )
            print(
                f"  Filtre de date {bounds} : "
                f"{n_excluded_date} oeuvre(s) hors periode, "
                f"{n_excluded_unparseable} avec date illisible -- exclues."
            )
        return works

    # ================================================================== #
    # Incremental save / resume                                          #
    # ================================================================== #
    @staticmethod
    def _load_existing_results(output_path: Path) -> tuple[list[dict], set[str]]:
        """
        Load an existing partial output (if any) and return
        (results_so_far, set_of_done_ids). Used for resuming.
        """
        if not output_path.exists():
            return [], set()
        try:
            with output_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return [], set()

            # A work counts as "done" only if NO construct is in API-error state
            # (known is None). known=False is a legitimate "model doesn't know it"
            # and is kept. Works hit by an API/limit error (known=None) are dropped
            # so they get re-annotated on resume instead of being silently skipped.
            def errored(r):
                return any(v is None for v in r.get("_known", {}).values())

            good = [r for r in data if "id" in r and not errored(r)]
            n_dropped = sum(1 for r in data if "id" in r and errored(r))
            if n_dropped:
                print(
                    f"  ↻ {n_dropped} œuvre(s) avec erreur API écartée(s) — seront refaites."
                )
            done_ids = {r["id"] for r in good}
            return good, done_ids
        except (json.JSONDecodeError, KeyError):
            print(f"  ⚠ Fichier de sortie existant illisible — on repart de zéro")
            return [], set()

    @staticmethod
    def _save_results(results: list[dict], output_path: Path) -> None:
        """Atomically write the results list to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        tmp.replace(output_path)  # atomic on the same filesystem

    # ================================================================== #
    # Public API                                                         #
    # ================================================================== #
    def process_csv(
        self,
        csv_path: str | Path,
        output_path: str | Path,
        id_col: str = "roland_id",
        title_col: str = "babel_title",
        author_col: str = "babel_author",
        date_col: str = "babel_date",
        per_ingredient_dir: str | Path | None = None,
        limit: int | None = None,
        save_every: int = 50,
        resume: bool = True,
        min_date: int | None = None,
        max_date: int | None = None,
        country_col: str | None = None,
        country: str | None = None,
    ) -> list[dict]:
        """
        Process all works in a CSV and write a JSON output compatible with
        annotator.py (same `<ING_ID>_Role` columns).

        Parameters
        ----------
        csv_path : path to the CSV
        output_path : where to write the JSON output
        id_col, title_col, author_col, date_col : column names in the CSV
        per_ingredient_dir : optional directory for per-ingredient exports
        limit : if set, only process the first N works (useful for testing)
        save_every : write the partial result to disk every N works
        resume : if True and the output file exists, skip works already done
        min_date, max_date : if set, restrict to works whose date is within
                             [min_date, max_date] (inclusive)
        """
        output_path = Path(output_path)
        works = self.load_works_csv(
            csv_path,
            id_col,
            title_col,
            author_col,
            date_col,
            min_date=min_date,
            max_date=max_date,
            country_col=country_col,
            country=country,
        )
        if limit is not None:
            works = works[:limit]

        n_total = len(works)

        # ── Resume support ──
        results: list[dict] = []
        done_ids: set[str] = set()
        if resume:
            results, done_ids = self._load_existing_results(output_path)
            if done_ids:
                print(
                    f"  ↻ Reprise : {len(done_ids)} œuvre(s) déjà annotée(s), "
                    f"on saute ces entrées."
                )

        todo = [w for w in works if w["id"] not in done_ids]
        n_todo = len(todo)

        print(f"\nAnnotation latente :")
        print(f"  Œuvres dans le CSV : {n_total}")
        print(f"  Déjà faites        : {len(done_ids)}")
        print(f"  À faire            : {n_todo}")
        print(f"  Requêtes API à venir : {n_todo * len(self.ingredients)}")
        print(f"  Sauvegarde tous les {save_every} œuvres\n")

        if n_todo == 0:
            print("  ✓ Tout est déjà annoté.")
        else:
            # If the API goes down (limit/auth/network), every construct of a work
            # comes back as an error (known=None). After a few such works in a row,
            # the API is clearly down — stop instead of churning thousands of dead
            # calls. The finally-block guarantees progress is saved "at the spot"
            # on ANY exit (API stop, Ctrl-C, unexpected crash), so re-running the
            # same command resumes exactly where it left off.
            max_consecutive_fail = 5
            consecutive_fail = 0
            try:
                for i, w in enumerate(todo, 1):
                    row = self.annotate_work(
                        work_id=w["id"],
                        title=w["title"],
                        author=w["author"],
                        date=w["date"],
                        index=len(done_ids) + i,
                        total=n_total,
                    )
                    results.append(row)

                    known_vals = list(row.get("_known", {}).values())
                    if known_vals and all(v is None for v in known_vals):
                        consecutive_fail += 1
                    else:
                        consecutive_fail = 0
                    if consecutive_fail >= max_consecutive_fail:
                        print(
                            f"\n  ⛔ {consecutive_fail} œuvres consécutives en échec API "
                            f"— l'API semble indisponible (quota/auth/réseau). Arrêt.\n"
                            f"     Relance la même commande pour reprendre ici."
                        )
                        break

                    # Incremental save
                    if i % save_every == 0:
                        self._save_results(results, output_path)
                        print(f"  💾 Sauvegarde intermédiaire ({len(results)} œuvres)")
            except KeyboardInterrupt:
                print("\n  ⛔ Interruption clavier — sauvegarde de la progression…")
            finally:
                # Always persist what we have, whatever caused the exit.
                self._save_results(results, output_path)
                print(f"\n  ✓ {len(results)} œuvres sauvegardées → {output_path}")

        # ── Summary ──
        n_known = sum(
            sum(1 for v in r.get("_known", {}).values() if v is True) for r in results
        )
        n_na = sum(
            sum(1 for v in r.get("_known", {}).values() if v is False) for r in results
        )
        n_err = sum(
            sum(1 for v in r.get("_known", {}).values() if v is None) for r in results
        )
        print(f"\nRésumé :")
        print(f"  Réponses 'known=True'        : {n_known}")
        print(f"  Réponses 'known=False' (NA)  : {n_na}")
        print(f"  Erreurs API (à relancer)     : {n_err}")

        if per_ingredient_dir:
            self.export_per_ingredient(results, per_ingredient_dir)

        return results

    # ================================================================== #
    # Per-ingredient export (compatible with annotator.py)               #
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
                    "title": r.get("title"),
                    "author": r.get("author"),
                    "Role": r.get(key),
                    "known": r.get("_known", {}).get(ing.id),
                    "explanation": r.get("_explanations", {}).get(ing.id),
                }
                for r in results
            ]
            out = per_ingredient_dir / f"{ing.id}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(slim, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {ing.id} ({ing.name}) → {out}")
