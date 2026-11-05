"""Annotate bibliographic metadata using model knowledge."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation import load_ingredients  # noqa: E402
from annotator_latent import (  # noqa: E402
    LatentLiteraryAnnotator,
    build_rubric,
    build_instructions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate French literary works from title+author only (Dubourg et al. method).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the CSV (babel_roland_matches_strict.csv or babel_french.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the JSON output is written.",
    )
    parser.add_argument(
        "--output-name",
        default="latent_annotations.json",
        help="Filename for the consolidated output (default: %(default)s).",
    )
    parser.add_argument(
        "--per-ingredient-dir",
        type=Path,
        default=None,
        help="Optional directory for per-ingredient exports.",
    )
    parser.add_argument(
        "--ingredients",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "themes.txt",
        help="Path to the themes file (default: %(default)s).",
    )
    # Column names — defaults match babel_roland_matches_strict.csv;
    # for babel_french.csv pass: --id-col title_id --title-col title etc.
    parser.add_argument(
        "--id-col",
        default="roland_id",
        help="ID column in the CSV (default: %(default)s). "
        "Use 'title_id' for babel_french.csv.",
    )
    parser.add_argument(
        "--title-col",
        default="babel_title",
        help="Title column (default: %(default)s). Use 'title' for babel_french.csv.",
    )
    parser.add_argument(
        "--author-col",
        default="babel_author",
        help="Author column (default: %(default)s). Use 'author' for babel_french.csv.",
    )
    parser.add_argument(
        "--date-col",
        default="babel_date",
        help="Date column (default: %(default)s). Use 'date' for babel_french.csv.",
    )
    parser.add_argument(
        "--country-col",
        default=None,
        help="Optional country column to filter on "
        "(e.g. 'country_harmonized' in babel_full_annotated.csv).",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Only annotate works whose --country-col equals this "
        "value (e.g. 'India', 'Japan').",
    )
    parser.add_argument(
        "--tradition",
        default="French",
        help="Literary tradition injected in the scoring prompt "
        "(default: %(default)s). Use 'Indian', 'Japanese', etc. "
        "so the model is not told a non-French work is French.",
    )
    parser.add_argument(
        "--no-justification",
        action="store_true",
        help="Score only (no explanation) — ~5x cheaper on output "
        "tokens. Drops the per-ingredient justification.",
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help="Score ALL ingredients in ONE API call per work instead of "
        "one call per (work, ingredient). ~1/N the calls and the "
        "rubric is sent once -> much cheaper. Claude backend only.",
    )
    parser.add_argument(
        "--medium",
        default="book",
        choices=["book", "film"],
        help="Wording of the work identifier (default: %(default)s). "
        "'book' -> \"by <author>, written in\"; "
        "'film' -> \"directed by <director>, released in\".",
    )

    parser.add_argument(
        "--backend",
        choices=["openai", "claude"],
        default="openai",
        help="Annotation backend (default: openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Default: gpt-4o-2024-08-06 (openai) / "
        "claude-opus-4-8 (claude). Claude aliases: haiku/sonnet/opus.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Best-effort OpenAI seed for reproducibility.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N works (useful for testing).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Write a partial result to disk every N works (default: %(default)s).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing output file and start fresh "
        "(default: resume from where a previous run stopped).",
    )
    parser.add_argument(
        "--min-date",
        type=int,
        default=None,
        help="Only annotate works dated on or after this year (e.g. 1500).",
    )
    parser.add_argument(
        "--max-date",
        type=int,
        default=None,
        help="Only annotate works dated on or before this year (e.g. 2000).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the latent-annotation CLI."""
    args = _parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    key_var = "ANTHROPIC_API_KEY" if args.backend == "claude" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_var)
    if not api_key:
        raise SystemExit(
            f"{key_var} not found. Add it to "
            f"{PROJECT_ROOT / '.env'} or export it before running."
        )

    model = args.model or (
        "claude-opus-4-8" if args.backend == "claude" else "gpt-4o-2024-08-06"
    )

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")
    if args.input.suffix.lower() != ".csv":
        raise SystemExit(f"Expected a .csv file, got: {args.input}")

    ingredients = load_ingredients(args.ingredients)
    print(f"{len(ingredients)} ingrédient(s) chargé(s) depuis {args.ingredients}")
    for ing in ingredients:
        print(f"  - {ing.id} : {ing.name}")
    print()
    print(
        f"Backend: {args.backend}  Model: {model}  "
        f"temperature={args.temperature}  seed={args.seed}"
    )
    print(f"Tradition (prompt): {args.tradition} literary works")
    print(f"Input CSV : {args.input}")
    print(
        f"  colonnes : id={args.id_col} title={args.title_col} "
        f"author={args.author_col} date={args.date_col}"
    )
    if args.country:
        print(f"  filtre pays : {args.country_col} == '{args.country}'")
    print(f"Output dir: {args.output_dir}")
    if args.per_ingredient_dir:
        print(f"Per-ingredient dir: {args.per_ingredient_dir}")
    if args.limit:
        print(f"⚠ LIMIT: only processing first {args.limit} works")
    if args.min_date or args.max_date:
        lo = args.min_date if args.min_date is not None else "−∞"
        hi = args.max_date if args.max_date is not None else "+∞"
        print(f"Date filter: [{lo}, {hi}]")
    print(f"Save every: {args.save_every} works    Resume: {not args.no_resume}")
    print()

    if args.medium == "film":
        creator_prefix, date_verb = "directed by", "released in"
        rubric = build_rubric(args.tradition, work_noun="films", period="20th–21st c.")
        instructions = build_instructions(
            creator="director", date_word="release date", work="film", works="films"
        )
    else:
        creator_prefix, date_verb = "by", "written in"
        rubric = build_rubric(args.tradition)
        instructions = build_instructions()
    common = dict(
        ingredients=ingredients,
        model=model,
        temperature=args.temperature,
        api_key=api_key,
        seed=args.seed,
        rubric=rubric,
        instructions=instructions,
        creator_prefix=creator_prefix,
        date_verb=date_verb,
        justify=not args.no_justification,
    )
    if args.backend == "claude":
        from annotator_latent_claude import ClaudeLatentAnnotator

        annotator = ClaudeLatentAnnotator(**common, group=args.group)
        if args.group:
            print(
                "⚡ Mode GROUPÉ : 1 appel/œuvre pour tous les construits (≈8x moins d'appels)."
            )
    else:
        if args.group:
            print(
                "⚠ --group n'est pour l'instant supporté qu'avec --backend claude ; "
                "exécution en mode non groupé."
            )
        annotator = LatentLiteraryAnnotator(**common)

    output_path = args.output_dir / args.output_name
    annotator.process_csv(
        csv_path=args.input,
        output_path=output_path,
        id_col=args.id_col,
        title_col=args.title_col,
        author_col=args.author_col,
        date_col=args.date_col,
        per_ingredient_dir=args.per_ingredient_dir,
        limit=args.limit,
        save_every=args.save_every,
        resume=not args.no_resume,
        min_date=args.min_date,
        max_date=args.max_date,
        country_col=args.country_col,
        country=args.country,
    )


if __name__ == "__main__":
    main()
