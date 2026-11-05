"""Bayesian author-intercept models of text-based romantic signalling."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORPORA = {
    "roland_fr": "metadata_roland_annotations.csv",
    "novels_uk": "metadata_english_works.csv",
    "theatre_fr": "metadata_theatre_fr.csv",
    "theatre_uk": "metadata_theatre_uk.csv",
}
LABELS = {
    "roland_fr": "French books (Roland)",
    "novels_uk": "English books",
    "theatre_fr": "French theatre",
    "theatre_uk": "English theatre",
}


def load_corpus(root, corpus):
    """Use the same metadata dates and author groups as four_corpora.py."""
    directory = root / "data/annotations/main" / corpus
    paths = [directory / "annotations_gpt.json", directory / CORPORA[corpus]]
    records = json.loads(paths[0].read_text())
    df = pd.DataFrame(records)
    metadata = pd.read_csv(paths[1], dtype={"id": str})
    for frame in (df, metadata):
        if frame.id.isna().any() or frame.id.astype(str).str.strip().eq("").any():
            raise ValueError(f"{corpus}: missing ID")
        frame["id"] = frame.id.astype(str)
        if frame.id.duplicated().any():
            raise ValueError(f"{corpus}: duplicate ID")
    df = df.drop(columns=["date", "author"], errors="ignore").merge(
        metadata[["id", "author", "date"]],
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if df._merge.ne("both").any():
        raise ValueError(f"{corpus}: missing metadata")
    for source, target in [("date", "year"), ("TH_03_Role", "score")]:
        values = df[source].replace(r"^\s*$", np.nan, regex=True)
        if values.map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f"{corpus}: boolean {target}")
        df[target] = pd.to_numeric(values, errors="raise")
        valid = df[target].dropna()
        if not np.isfinite(valid).all() or not valid.eq(np.floor(valid)).all():
            raise ValueError(f"{corpus}: {target} must be finite integers")
    if not df.score.dropna().between(0, 6).all():
        raise ValueError(f"{corpus}: scores outside 0–6")
    if not df.year.dropna().between(1000, 2029).all():
        raise ValueError(f"{corpus}: years outside 1000–2029")
    excluded = df.loc[df[["year", "score"]].isna().any(axis=1), "id"].tolist()
    df = df.dropna(subset=["year", "score"]).copy()
    if df.empty:
        raise ValueError(f"{corpus}: no valid observations")
    df["year"] = df.year.astype(int)
    df["score"] = df.score.astype(int)
    df["centered_year"] = (df.year - df.year.mean()) / 100
    df["score_categories"] = pd.Categorical(df.score, categories=range(7), ordered=True)
    # Preserve the original four_corpora.py treatment of missing authors.
    authors = df.author.replace(["Auteur inconnu", "Unknown Author"], np.nan)
    unknown_count = int(authors.isna().sum())
    df["author"] = authors.fillna("unknown_" + df.id)
    df = df[
        ["id", "author", "year", "score", "centered_year", "score_categories"]
    ].copy()
    df.attrs["provenance"] = {
        "corpus": corpus,
        "construct": "TH_03",
        "source_rows": len(records),
        "included_rows": len(df),
        "excluded_ids": excluded,
        "unknown_authors": unknown_count,
        "year_min": int(df.year.min()),
        "year_max": int(df.year.max()),
        "sources": [
            {
                "path": str(p.relative_to(root)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        ],
    }
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--twolines-script", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Resample instead of loading saved posterior draws",
    )
    args = parser.parse_args()
    if min(args.draws, args.tune, args.chains, args.cores) < 1:
        parser.error("Sampling settings must be positive")
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "results/signaling"
    for folder in ("data", "tables", "models", "figures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    # Reuse the verified Bayesian, two-lines, GP, table and figure functions.
    spec = importlib.util.spec_from_file_location(
        "signaling_analysis", root / "analysis/latent_corpora.py"
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.OUTPUT = output
    analysis.REFIT = args.refit
    analysis.ANALYSES = {f"{corpus}_TH_03": (corpus, "TH_03") for corpus in CORPORA}
    analysis.LABELS = {f"{corpus}_TH_03": label for corpus, label in LABELS.items()}
    analysis.DRAWS, analysis.TUNE = args.draws, args.tune
    analysis.CHAINS, analysis.CORES = args.chains, args.cores
    data = {f"{corpus}_TH_03": load_corpus(root, corpus) for corpus in CORPORA}
    manifest = {
        "construct": "TH_03",
        "measurement": "text-based",
        "groups": {},
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "date_rule": "Metadata dates, exactly as in four_corpora.py; no common-period truncation",
        "ordinal_curve": "Quadratic at author effect zero, not selected by LOO",
        "unit": "Original annotation rows; repeated Roland source books are retained as in four_corpora.py",
        "reference_script_sha256": hashlib.sha256(
            (root / "analysis/four_corpora.py").read_bytes()
        ).hexdigest(),
        "shared_analysis_sha256": hashlib.sha256(
            (root / "analysis/latent_corpora.py").read_bytes()
        ).hexdigest(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bambi": analysis.bmb.__version__,
            "arviz": analysis.az.__version__,
        },
    }
    for name, df in data.items():
        manifest["groups"][name] = df.attrs["provenance"]
        df.drop(columns="score_categories").to_csv(
            output / "data" / f"{name}.csv", index=False
        )
        print(f"{name}: n={len(df)}, years={df.year.min()}–{df.year.max()}", flush=True)
    manifest_path = output / "data/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if args.prepare_only:
        return
    script = (
        args.twolines_script.resolve()
        if args.twolines_script
        else analysis.download_twolines(output / "data")
    )
    manifest["twolines_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    results = {}
    for name, df in data.items():
        results[name] = analysis.analyse_corpus(name, script, df)
        analysis.export_tables(results)
    analysis.plot_trend_grid(
        results,
        "ordinal",
        "Quadratic ordinal expected score (author effect = 0)",
        "Text-based romantic signaling",
        "figure_ordinal",
    )
    analysis.plot_trend_grid(
        results,
        "gp",
        "Gaussian process predicted mean score",
        "Text-based romantic signaling (Gaussian process)",
        "figure_gp",
    )
    analysis.plot_probability_grid(results)
    print(f"Wrote results to {output}", flush=True)


if __name__ == "__main__":
    main()
