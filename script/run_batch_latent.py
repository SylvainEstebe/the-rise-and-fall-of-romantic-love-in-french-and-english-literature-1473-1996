#!/usr/bin/env python3
"""Submit and collect Claude batch annotations of bibliographic metadata."""

import argparse, os, sys, time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "script")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.annotation import load_ingredients  # noqa: E402
from annotator_latent import (
    LatentLiteraryAnnotator,
    build_rubric,  # noqa: E402
    build_instructions,
    build_grouped_prompt,
    build_user_message,
)
from annotator_latent_claude import build_grouped_tool  # noqa: E402
from src.annotation.annotation_claude import MODEL_ALIASES  # noqa: E402
from anthropic import Anthropic  # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # noqa: E402
from anthropic.types.messages.batch_create_params import Request  # noqa: E402

NOJUST = (
    "Provide a brief explanation (2–4 sentences), and return your judgment "
    "in the structured format.",
    "Return your judgment in the structured format.",
)


def _args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-name", default="latent_annotations.json")
    p.add_argument("--ingredients", type=Path, default=ROOT / "prompts/themes_8.txt")
    p.add_argument("--id-col", default="title_id")
    p.add_argument("--title-col", default="title")
    p.add_argument("--author-col", default="author")
    p.add_argument("--date-col", default="date")
    p.add_argument("--country-col", default=None)
    p.add_argument("--country", default=None)
    p.add_argument("--tradition", default="British")
    p.add_argument("--medium", default="book", choices=["book", "film"])
    p.add_argument("--model", default="opus")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--min-date", type=int, default=None)
    p.add_argument("--max-date", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--poll", type=int, default=30, help="seconds between status polls")
    return p.parse_args()


def main():
    a = _args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit(
        "ANTHROPIC_API_KEY manquant (.env)."
    )
    model = MODEL_ALIASES.get(a.model, a.model)
    ings = load_ingredients(a.ingredients)

    if a.medium == "film":
        creator_prefix, date_verb = "directed by", "released in"
        rubric = build_rubric(a.tradition, work_noun="films", period="20th–21st c.")
        instr = build_instructions(
            creator="director", date_word="release date", work="film", works="films"
        )
    else:
        creator_prefix, date_verb = "by", "written in"
        rubric = build_rubric(a.tradition)
        instr = build_instructions()
    instr = instr.replace(*NOJUST)  # grouped = score-only
    grouped_system = build_grouped_prompt(ings, rubric, instr)
    tool = build_grouped_tool(ings)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = a.output_dir / a.output_name
    results, done = LatentLiteraryAnnotator._load_existing_results(
        out_path
    )  # error-aware resume
    works = LatentLiteraryAnnotator.load_works_csv(
        a.input,
        a.id_col,
        a.title_col,
        a.author_col,
        a.date_col,
        min_date=a.min_date,
        max_date=a.max_date,
        country_col=a.country_col,
        country=a.country,
    )
    if a.limit:
        works = works[: a.limit]
    by_id = {w["id"]: w for w in works}
    todo = [w for w in works if w["id"] not in done]

    print(f"Modèle {model} | GROUPÉ + BATCH (−50%) | {len(ings)} construits")
    print(f"Œuvres CSV={len(works)}  déjà faites={len(done)}  à faire={len(todo)}")

    client = Anthropic(api_key=key)
    state = a.output_dir / ".batch_id"

    if state.exists():
        batch_id = state.read_text().strip()
        print(f"↻ Reprise du batch en cours : {batch_id} (pas de resoumission).")
    else:
        if not todo:
            print("✓ Tout est déjà annoté.")
            return
        reqs = [
            Request(
                custom_id=w["id"],
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=a.max_tokens,
                    system=grouped_system,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[
                        {
                            "role": "user",
                            "content": build_user_message(
                                w["title"],
                                w["author"],
                                w["date"],
                                creator_prefix,
                                date_verb,
                            ),
                        }
                    ],
                ),
            )
            for w in todo
        ]
        if len(reqs) > 100_000:
            sys.exit("Plus de 100 000 requêtes : découper le batch (limite Anthropic).")
        batch = client.messages.batches.create(requests=reqs)
        batch_id = batch.id
        state.write_text(batch_id)
        print(
            f"📦 Batch soumis : {batch_id} ({len(reqs)} requêtes). Coût −50%. "
            f"Généralement < 1 h (max 24 h)."
        )

    # ---- poll ----
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(
            f"  … {b.processing_status} : processing={rc.processing} "
            f"succeeded={rc.succeeded} errored={rc.errored}"
        )
        time.sleep(a.poll)

    # ---- retrieve + parse ----
    n_ok = n_err = 0
    for res in client.messages.batches.results(batch_id):
        wid = res.custom_id
        if wid in done:  # already saved on a previous run → skip (no duplicate)
            continue
        if res.result.type != "succeeded":
            n_err += 1
            continue
        parsed = None
        for block in res.result.message.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and block.name == tool["name"]
            ):
                parsed = dict(block.input or {})
        if parsed is None:
            n_err += 1
            continue
        w = by_id.get(wid, {})
        row = {
            "id": wid,
            "book_id": wid,
            "date": w.get("date"),
            "title": w.get("title"),
            "author": w.get("author"),
            "_explanations": {},
            "_known": {},
        }
        for ing in ings:
            known = parsed.get(f"{ing.id}_known")
            row[f"{ing.id}_Role"] = parsed.get(f"{ing.id}_score") if known else None
            row["_known"][ing.id] = known
            row["_explanations"][ing.id] = None
        results.append(row)
        done.add(wid)
        n_ok += 1

    LatentLiteraryAnnotator._save_results(results, out_path)
    if state.exists():
        state.unlink()
    print(
        f"\n✓ {n_ok} ajoutées | {n_err} en erreur (refaites au prochain run) "
        f"| total {len(results)} → {out_path}"
    )


if __name__ == "__main__":
    main()
