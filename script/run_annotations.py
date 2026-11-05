"""Annotate exact text batches with the OpenAI or Claude client."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation import LiteraryAnnotator, load_ingredients  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate French literary texts via the OpenAI API."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Either a parent directory (subfolders/files = batches) or a "
            "single .json/.txt file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-batch JSON outputs are written.",
    )
    parser.add_argument(
        "--per-ingredient-dir",
        type=Path,
        default=None,
        help="Optional directory for per-ingredient exports (only used when "
        "--input is a directory).",
    )
    parser.add_argument(
        "--ingredients",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "themes.txt",
        help="Path to the themes file (one 'ID | Name | Description' per line).",
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
        help="Model id. Default: gpt-5.4-2026-03-05 (openai) / "
        "claude-opus-4-8 (claude). Claude aliases: haiku/sonnet/opus.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = deterministic-ish).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Best-effort OpenAI seed for reproducibility across runs.",
    )
    parser.add_argument(
        "--tradition",
        default="French",
        help="Literary tradition injected in the system prompt (default: French). "
        "Use 'English' for the English corpus so texts are not labelled French.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint for --backend openai. "
        "DeepSeek: https://api.deepseek.com ; local vLLM/Ollama: http://localhost:8000/v1.",
    )
    parser.add_argument(
        "--rubric-file",
        type=Path,
        default=None,
        help="Text file whose content REPLACES the default rubric verbatim "
        "(overrides --tradition). For prompt-calibration experiments.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the annotation CLI."""
    args = _parse_args()

    # Load .env from project root (silent no-op if missing).
    load_dotenv(PROJECT_ROOT / ".env")
    key_var = "ANTHROPIC_API_KEY" if args.backend == "claude" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_var)
    if not api_key:
        raise SystemExit(
            f"{key_var} not found. Add it to "
            f"{PROJECT_ROOT / '.env'} or export it before running."
        )

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    model = args.model or (
        "claude-opus-4-8" if args.backend == "claude" else "gpt-5.4-2026-03-05"
    )

    ingredients = load_ingredients(args.ingredients)
    print(f"{len(ingredients)} ingrédient(s) chargé(s) depuis {args.ingredients}")
    for ing in ingredients:
        print(f"  - {ing.id} : {ing.name}")
    print()
    print(
        f"Backend: {args.backend}  Model: {model}  "
        f"temperature={args.temperature}  seed={args.seed}"
    )
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    if args.per_ingredient_dir:
        print(f"Per-ingredient dir: {args.per_ingredient_dir}")
    print()

    # Swap the hardcoded "French" tradition in the system prompt when requested,
    # or replace the whole rubric with a file (prompt-calibration experiments).
    from src.annotation.annotator import DEFAULT_RUBRIC

    if args.rubric_file:
        rubric = args.rubric_file.read_text(encoding="utf-8")
        print(f"Rubric (prompt): {args.rubric_file}")
    else:
        rubric = DEFAULT_RUBRIC.replace(
            "French literary texts", f"{args.tradition} literary texts"
        )
        print(f"Tradition (prompt): {args.tradition} literary texts")

    if args.backend == "claude":
        from src.annotation.annotation_claude import ClaudeAnnotator

        annotator = ClaudeAnnotator(
            ingredients=ingredients,
            model=model,
            temperature=args.temperature,
            api_key=api_key,
            seed=args.seed,
            rubric=rubric,
        )
    else:
        if args.base_url:
            print(f"Endpoint OpenAI-compatible : {args.base_url}")
        annotator = LiteraryAnnotator(
            ingredients=ingredients,
            model=model,
            temperature=args.temperature,
            api_key=api_key,
            seed=args.seed,
            rubric=rubric,
            base_url=args.base_url,
        )

    if args.input.is_dir():
        annotator.process_directory(
            input_dir=args.input,
            output_dir=args.output_dir,
            per_ingredient_dir=args.per_ingredient_dir,
        )
    else:
        # Single file: write to <output-dir>/<input-stem>.json
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.input.stem}.json"
        annotator.process_batch(args.input, output_path)


if __name__ == "__main__":
    main()
