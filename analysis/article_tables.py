"""Export the numerical tables and figure index used by the manuscript."""

from pathlib import Path

import pandas as pd

from four_corpora import CORPORA, az, load_corpus
from latent_corpora import ANALYSES, load_corpus as load_babel
from text_signaling import load_corpus as load_signaling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/article"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary = {key: load_corpus(key) for key in CORPORA}
    coverage = []
    authors = []
    availability = []
    for key, d in primary.items():
        counts = d.groupby("author").size()
        coverage.append(
            dict(corpus=key, n=len(d), year_min=d.year.min(), year_max=d.year.max())
        )
        authors.append(
            dict(
                corpus=key,
                author_groups=len(counts),
                unknown_author_groups=int(
                    counts.index.str.startswith("unknown_").sum()
                ),
                median_records=counts.median(),
                min_records=counts.min(),
                max_records=counts.max(),
            )
        )
        for construct, frame in [("ING_38", d), ("TH_03", load_signaling(ROOT, key))]:
            availability.append(dict(corpus=key, construct=construct, n=len(frame)))
    for key, (corpus, construct) in ANALYSES.items():
        availability.append(
            dict(corpus=corpus, construct=construct, n=len(load_babel(key)))
        )
    pd.DataFrame(coverage).to_csv(OUT / "table_1.csv", index=False)
    pd.DataFrame(availability).pivot(
        index="construct", columns="corpus", values="n"
    ).to_csv(OUT / "table_s4.csv")
    pd.DataFrame(authors).to_csv(OUT / "table_s6.csv", index=False)

    coefficients = []
    for corpus in CORPORA:
        for shape in ("linear", "quadratic"):
            idata = az.from_netcdf(ROOT / f"results/primary/models/{corpus}_{shape}.nc")
            variables = ["centered_year", "1|author_sigma"]
            if shape == "quadratic":
                variables.insert(1, "I(centered_year ** 2)")
            summary = (
                az.summary(idata, var_names=variables)
                .rename_axis("parameter")
                .reset_index()
            )
            summary.insert(0, "shape", shape)
            summary.insert(0, "corpus", corpus)
            coefficients.append(summary)
            idata.close()
    pd.concat(coefficients).to_csv(OUT / "primary_coefficients.csv", index=False)

    comparisons = pd.read_csv(
        ROOT / "results/primary/tables/model_comparison.csv"
    ).pivot(index="corpus", columns="parametre", values="resultat")
    loo = []
    for corpus in CORPORA:
        row = comparisons.loc[corpus]
        value = lambda name: float(row[name])
        loo.append(
            dict(
                corpus=corpus,
                linear_elpd=value("lineaire.elpd_loo"),
                linear_se=value("lineaire.se"),
                quadratic_elpd=value("quadratique.elpd_loo"),
                quadratic_se=value("quadratique.se"),
                linear_p_loo=value("lineaire.p_loo"),
                quadratic_p_loo=value("quadratique.p_loo"),
                delta_elpd=value("quadratique.elpd_loo") - value("lineaire.elpd_loo"),
                delta_se=max(value("lineaire.dse"), value("quadratique.dse")),
            )
        )
    loo = pd.DataFrame(loo)
    loo.to_csv(OUT / "table_s7.csv", index=False)
    lines = pd.read_csv(ROOT / "results/primary/tables/twolines_with_holm.csv")
    loo[["corpus", "delta_elpd", "delta_se"]].merge(
        lines, on="corpus", validate="one_to_one"
    ).to_csv(OUT / "table_2.csv", index=False)

    validation = pd.read_csv(ROOT / "results/validation/validation_ci.csv")
    select = [
        ("arman vs sylvain (human-human)", "ING_38"),
        ("humans_mean vs GPT", "ING_38"),
        ("arman vs GPT", "TH_03"),
    ]
    pd.concat(
        [
            validation[
                (validation.comparison == comparison)
                & (validation.construct == construct)
            ]
            for comparison, construct in select
        ]
    ).to_csv(OUT / "table_s5.csv", index=False)
    decomposition = pd.read_csv(ROOT / "results/courtship/observed_contrasts.csv")
    decomposition.query(
        'variant == "all" and threshold == 1 and early == "1800–1899"'
    ).to_csv(OUT / "table_s8.csv", index=False)
    figures = [
        ("1", "primary/figures/figure_ordinal"),
        ("S1", "primary/figures/figure_s1_histograms"),
        ("S2", "signaling/figures/figure_ordinal"),
        ("S3", "latent/figures/figure_ordinal"),
        *[
            (f"S{n}{part}", f"courtship/doc_assets/figure_s{n}{part}")
            for n in (4, 5, 6)
            for part in ("a", "b")
        ],
    ]
    rows = []
    for figure, stem in figures:
        paths = {ext: f"results/{stem}.{ext}" for ext in ("png", "pdf")}
        for path in paths.values():
            assert (ROOT / path).is_file(), path
        rows.append(dict(figure=figure, **paths))
    pd.DataFrame(rows).to_csv(OUT / "figures.csv", index=False)
    # Keep historical numbered filenames, but expose stable descriptive names.
    # Their old supplementary numbers differ from the current manuscript.
    import shutil

    aliases = {
        "table_s4.csv": "construct_coverage.csv",
        "table_s5.csv": "human_validation.csv",
        "table_s6.csv": "author_groups.csv",
        "table_s7.csv": "model_comparison.csv",
        "table_s8.csv": "century_contrasts.csv",
    }
    for original, descriptive in aliases.items():
        shutil.copyfile(OUT / original, OUT / descriptive)
    shutil.copyfile(
        ROOT / "results/submission/descriptive_statistics.csv",
        OUT / "descriptive_statistics.csv",
    )
    table_index = [
        ("1", "Primary corpus coverage", "results/article/table_1.csv", "analysis/article_tables.py"),
        ("2", "Model comparison and two-lines tests", "results/article/table_2.csv", "analysis/article_tables.py"),
        ("S1", "Corpus descriptions and analytical samples", "data/README.md", "documentation"),
        ("S2", "Construct descriptive statistics", "results/article/descriptive_statistics.csv", "analysis/submission_statistics.py"),
        ("S3", "Scoring rubric", "docs/ANNOTATION.md", "documentation"),
        ("S4", "Construct definitions", "prompts/themes.txt", "annotation definitions"),
        ("S5", "Construct coverage", "results/article/construct_coverage.csv", "analysis/article_tables.py"),
        ("S6", "Human validation", "results/article/human_validation.csv", "analysis/validation.py + analysis/article_tables.py"),
        ("S7", "Author groups", "results/article/author_groups.csv", "analysis/article_tables.py"),
        ("S8", "Ordinal model comparison", "results/article/model_comparison.csv", "analysis/article_tables.py"),
        ("S9", "Century contrasts", "results/article/century_contrasts.csv", "analysis/descriptive.py + analysis/article_tables.py"),
    ]
    for _, _, path, _ in table_index:
        assert (ROOT / path).is_file(), path
    pd.DataFrame(table_index, columns=["manuscript_table", "description", "path", "generated_by"]).to_csv(
        OUT / "tables.csv", index=False
    )
    print(f"Exported tables, {len(table_index)} table references and {len(figures)} figure references to {OUT}")


if __name__ == "__main__":
    main()
