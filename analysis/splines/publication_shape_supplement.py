#!/usr/bin/env python3
"""Table S10, Figure S7, and French-book length sensitivity.

Run with the pinned environment used for ordinal_shape_robustness.py:
  python analysis/splines/publication_shape_supplement.py --stage all
Stages: length (new full fits), publication (saved fits only), report (length comparison).
No unadjusted model is resampled. Checkpoints allow interrupted length fits to resume.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

# isort: off
# The helper configures Matplotlib before ArviZ imports its plotting backends.
import ordinal_shape_robustness as a
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
# isort: on

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/splines/affiliation"
OUT = ROOT / "results/splines/robustness"
LENGTH = OUT / "length_adjusted"
LABELS = dict(
    zip(
        a.DATASETS,
        [
            "French books",
            "English books",
            "French theatre",
            "English theatre",
            "French Babel",
        ],
    )
)
MODELS = list(a.FORMULAS)
ADJUSTED = ["quadratic", "spline_4", "spline_5", "spline_6"]
QUANTITIES = [
    "contrast_1500_1650",
    "contrast_1650_1800",
    "contrast_1800_1950",
    "global_maximum_year",
    "decline_0.25_year",
]
HEADERS = [
    "1500→1650",
    "1650→1800",
    "1800→1950",
    "Maximum year; boundary %",
    "0.25-point decline year; never %",
]
NOTE = (
    "**Note.** ΔELPD is the nonnegative predictive-accuracy loss relative to the best of "
    "six models in each dataset; parentheses give its paired, observation-wise SE. "
    "“Within 2 SE” means ΔELPD < 2 SE, with the best model included by definition; "
    "this is descriptive, not a formal equivalence test. LOO includes exact corrections "
    "for the two flagged French Babel observations. Contrasts are later minus earlier "
    "expected scores; adjusted endpoints appear in parentheses, and — denotes no covered "
    "interval. All estimates are medians [95% equal-tailed credible intervals]. Maximum "
    "year is each draw’s global maximum on the observed yearly grid (earliest year if tied); "
    "boundary % combines maxima at the first and last observed years. Decline year is the "
    "first subsequent year at least 0.25 points below that maximum, conditional on crossing; "
    "never % means no crossing within the observed range. It is a threshold-crossing date, "
    "not the first negative slope. Expected scores are Σ k·P(score=k), with author "
    "deviations zero, using four chains × 4,000 draws from cumulative-logit models with "
    "author random intercepts and year centred at the corpus mean and divided by 100. "
    "A contrast interval including zero does not establish a plateau."
)


def model_label(name):
    return (
        name.capitalize() if not name.startswith("spline") else f"Spline df={name[-1]}"
    )


def load_fit(path):
    fit = az.from_netcdf(path)
    fit.load()
    fit.close()
    assert fit.posterior.sizes["chain"] == 4 and fit.posterior.sizes["draw"] == 4000
    return fit


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def length_data():
    data, mean, _ = a.prepare_data(SOURCE / "french_books/analysis_data.csv")
    linked = pd.read_csv(ROOT / "results/length/linked_data.csv")
    assert data.id.is_unique and linked.id.is_unique and set(data.id) == set(linked.id)
    linked = linked.set_index("id").loc[data.id]
    for col in ["year", "score", "author", "n_chapters"]:
        np.testing.assert_array_equal(
            data[col].astype(str).values, linked[col].astype(str).values
        )
    assert len(data) == 1879 and (data.n_chapters > 0).all()
    log2 = np.log2(data.n_chapters)
    data["centered_log2_length"] = log2 - log2.mean()
    return data, mean, float(2 ** log2.mean())


def run_length():
    LENGTH.mkdir(parents=True, exist_ok=True)
    data, mean, geometric = length_data()
    data.to_csv(LENGTH / "analysis_data.csv", index=False)
    manifest = {
        "source_data_sha256": checksum(SOURCE / "french_books/analysis_data.csv"),
        "n": len(data),
        "mean_year": mean,
        "geometric_mean_segment_count": geometric,
        "length_variable": "centered_log2_length = log2(n_chapters) - mean(log2(n_chapters))",
        "segment_provenance": "src/build_corpus/dataset_sampler.py: n_chapters = len(split_into_chapters(text)); approximately 15,000-character segments, not literary chapters",
        "draws": 4000,
        "chains": 4,
        "tune": 2000,
        "seed": 0,
        "retry_tune": 4000,
        "retry_target_accept": 0.95,
        "prediction_length": 0,
        "author_deviations": 0,
    }
    manifest_path = LENGTH / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, (
            "Checkpoint provenance changed"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    years = np.arange(data.year.min(), data.year.max() + 1)
    for name in ADJUSTED:
        formula = a.FORMULAS[name] + " + centered_log2_length"
        model = bmb.Model(formula, data, family="cumulative")
        model.build()
        (LENGTH / f"{name}_model.txt").write_text(str(model))
        fit, attempts = a.fit_checked(
            model,
            LENGTH,
            name,
            draws=4000,
            tune=2000,
            chains=4,
            cores=4,
            seed=0,
            target_accept=0.8,
        )
        assert not attempts[-1]["fails_requested_checks"], (
            f"{name}: unresolved MCMC checks"
        )
        expected = a.expected_scores(
            model, fit, years, mean, prediction_covariates={"centered_log2_length": 0}
        )
        # Check that the fitted formulae spline state is invariant to prediction batches.
        ix = np.array([0, len(years) // 3, len(years) // 2, len(years) - 1])
        check = a.expected_scores(
            model,
            fit,
            years[ix],
            mean,
            chunk_size=1,
            prediction_covariates={"centered_log2_length": 0},
        )
        np.testing.assert_allclose(expected[:, ix], check, atol=1e-10, rtol=1e-10)
        long, draws = a.long_shape_table(expected, years, name)
        long.to_csv(LENGTH / f"{name}_quantities.csv", index=False)
        draws.to_csv(LENGTH / f"{name}_shape_draws.csv", index=False)
        pd.DataFrame(
            {
                "year": years,
                "mean": expected.mean(0),
                "lower_95": np.quantile(expected, 0.025, axis=0),
                "upper_95": np.quantile(expected, 0.975, axis=0),
            }
        ).to_csv(LENGTH / f"{name}_curve.csv", index=False)
        print(f"COMPLETE adjusted {name}", flush=True)
        del fit, expected, model
        gc.collect()
    length_report()


def formatted(row, quantity):
    if not np.isfinite(row["median"]):
        return "—" + (
            f"; never {100 * row['proportion_never_within_range']:.1f}%"
            if quantity.startswith("decline")
            else ""
        )
    decimals = 2 if quantity.startswith("contrast") else 0
    text = f"{row['median']:.{decimals}f} [{row['lower_95']:.{decimals}f}, {row['upper_95']:.{decimals}f}]"
    if quantity.startswith("contrast") and bool(row["endpoints_adjusted"]):
        text += f" ({row['actual_start']:.0f}→{row['actual_end']:.0f})"
    if quantity == "global_maximum_year":
        text += f"; boundary {100 * (row['proportion_at_first_year'] + row['proportion_at_last_year']):.1f}%"
    if quantity.startswith("decline"):
        text += f"; never {100 * row['proportion_never_within_range']:.1f}%"
    return text


def publication_table():
    OUT.mkdir(parents=True, exist_ok=True)
    records, raw = [], []
    text = ["# Supplementary Table S10. Robustness to the shape of the time trend", ""]
    for dataset, label in LABELS.items():
        source = SOURCE / dataset
        data, mean, _ = a.prepare_data(source / "analysis_data.csv")
        loo = pd.read_csv(source / "loo_full.csv").set_index("model")
        long = pd.read_csv(source / "all_model_quantities.csv")
        # Preserve the original training data and stateful formulae design for linear predictions.
        model = bmb.Model(a.FORMULAS["linear"], data, family="cumulative")
        model.build()
        fit = load_fit(source / "linear.nc")
        years = np.arange(data.year.min(), data.year.max() + 1)
        expected = a.expected_scores(model, fit, years, mean)
        linear, draws = a.long_shape_table(expected, years, "linear")
        linear["dataset"] = dataset
        draws.to_csv(OUT / f"{dataset}_linear_shape_draws.csv", index=False)
        long = pd.concat([long, linear], ignore_index=True)
        raw.append(long)
        block = []
        for name in MODELS:
            row = {
                "Dataset": label,
                "Model": model_label(name),
                "ΔELPD vs best (SE)": f"{loo.loc[name, 'delta_elpd_loss_vs_best']:.2f} ({loo.loc[name, 'delta_elpd_se']:.2f})",
                "Within 2 SE": "yes" if loo.loc[name, "equivalent_2se"] else "no",
            }
            sub = long.loc[long.model.eq(name)].set_index("quantity")
            row.update(
                {h: formatted(sub.loc[q], q) for q, h in zip(QUANTITIES, HEADERS)}
            )
            block.append(row)
        block = pd.DataFrame(block)
        block.to_csv(OUT / f"Table_S10_{dataset}.csv", index=False)
        records.extend(block.to_dict("records"))
        text += [
            f"## {label} (n = {len(data):,}; {years[0]}–{years[-1]})",
            "",
            a.md_table(block.drop(columns="Dataset")),
            "",
        ]
        del model, fit, expected
        gc.collect()
        print(f"TABLE {dataset}", flush=True)
    pd.DataFrame(records).to_csv(OUT / "Table_S10.csv", index=False)
    numeric = pd.concat(raw, ignore_index=True)
    # Merge LOO values afresh so the newly computed linear rows also have comparison metadata.
    numeric = numeric.drop(
        columns=[c for c in numeric if c.startswith("without_flagged_")]
    )
    for dataset in LABELS:
        loo = pd.read_csv(SOURCE / dataset / "loo_full.csv").set_index("model")
        mask = numeric.dataset.eq(dataset)
        for col in loo.columns:
            numeric.loc[mask, col] = numeric.loc[mask, "model"].map(loo[col])
    numeric.to_csv(OUT / "Table_S10_numeric_long.csv", index=False)
    (OUT / "Table_S10.md").write_text("\n".join(text + [NOTE, ""]))


def publication_figure():
    # Match the article four-corpus figure: sans serif, blue means, pale grid,
    # lower-case panel letters, open top/right spines, common score and year axes.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 9.8), sharex=True, sharey=True)
    styles = {
        "spline_4": ("#D55E00", "-"),
        "spline_5": ("#009E73", ":"),
        "spline_6": ("#CC79A7", "-."),
    }
    selected = {}
    for i, (dataset, label) in enumerate(LABELS.items()):
        ax = axes.flat[i]
        source = SOURCE / dataset
        curves = pd.read_csv(source / "all_curves.csv")
        loo = pd.read_csv(source / "loo_full.csv").set_index("model")
        best = loo.loc[list(styles), "elpd_loo"].idxmax()
        selected[dataset] = best
        band = curves[curves.model.eq(best)]
        ax.fill_between(
            band.year,
            band.lower_95,
            band.upper_95,
            color=styles[best][0],
            alpha=0.14,
            linewidth=0,
        )
        for name, (color, ls) in styles.items():
            sub = curves[curves.model.eq(name)]
            ax.plot(sub.year, sub["mean"], color=color, ls=ls, lw=1.7)
        sub = curves[curves.model.eq("quadratic")]
        ax.plot(sub.year, sub["mean"], color="#888888", ls="--", lw=1.5)
        data = pd.read_csv(source / "analysis_data.csv")
        data["bin"] = 20 * np.floor(data.year / 20)
        bins = data.groupby("bin").agg(
            year=("year", "mean"),
            n=("score", "size"),
            mean=("score", "mean"),
            sd=("score", "std"),
        )
        bins = bins[bins.n.ge(10)].copy()
        bins["se"] = bins.sd / np.sqrt(bins.n)
        saved = pd.read_csv(source / "twenty_year_means.csv")
        np.testing.assert_allclose(
            bins[["year", "mean", "se"]], saved[["year", "mean", "se"]]
        )
        ax.errorbar(
            bins.year,
            bins["mean"],
            yerr=1.96 * bins.se,
            fmt="o",
            color="#0072B2",
            ms=3.6,
            mec="white",
            mew=0.45,
            elinewidth=0.9,
            capsize=2,
            zorder=5,
        )
        ax.set_title(f"{label} (n = {len(data):,})", pad=9)
        ax.text(
            -0.085,
            1.045,
            "abcde"[i],
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.04,
            f"Band: df = {best[-1]}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color="#555555",
        )
        ax.set_xlim(1470, 2005)
        # Leave room for the descriptive SE bar slightly below zero in English books.
        ax.set_ylim(-0.3, 6.2)
        ax.set_yticks(range(7))
        ax.set_xticks(range(1500, 2001, 100))
        ax.grid(color="#EEEEEE", lw=0.7)
        ax.set_axisbelow(True)
        # English theatre is the bottom data panel of the right-hand column.
        if i in (3, 4):
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Publication year")
    axes[2, 1].axis("off")
    handles = [
        Line2D([], [], color=c, ls=ls, lw=1.8, label=f"Spline df = {n[-1]}")
        for n, (c, ls) in styles.items()
    ]
    handles += [
        Line2D([], [], color="#888888", ls="--", lw=1.5, label="Quadratic"),
        Line2D(
            [],
            [],
            color="#0072B2",
            marker="o",
            ls="",
            ms=4,
            label="20-year mean ± 1.96 SE",
        ),
        Patch(
            facecolor="#999999",
            alpha=0.18,
            label="95% credible band for best\ndisplayed spline",
        ),
    ]
    axes[2, 1].legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=10,
        handlelength=3.2,
        labelspacing=1.1,
    )
    fig.supylabel("Expected courtship-importance score (0–6)", x=0.015)
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.96, bottom=0.065, hspace=0.30, wspace=0.16
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT / f"Figure_S7.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)
    (OUT / "Figure_S7_band_models.json").write_text(json.dumps(selected, indent=2))
    selections = "; ".join(f"{LABELS[d]}, df {n[-1]}" for d, n in selected.items())
    caption = (
        "**Supplementary Figure S7. Sensitivity of long-term courtship-importance trends "
        "to spline flexibility.** Panels show French books (a), English books (b), French "
        "theatre (c), English theatre (d), and French Babel (e). Blue points show means "
        "within calendar-aligned 20-year bins (20 × floor(year/20)), plotted at the mean "
        "publication year; error bars are ±1.96 standard errors, and only bins with at "
        "least 10 observations are shown. Solid orange, dotted green, and dash-dot purple "
        "curves show posterior mean expected scores from cubic B-splines with df 4, 5, "
        "and 6, respectively; the dashed grey curve shows the quadratic model. The light "
        "band is the pointwise 95% equal-tailed credible interval for the highest-ELPD "
        "spline among the three displayed models in each panel (" + selections + "). "
        "ELPD comparisons use PSIS-LOO with exact refits for flagged observations. Each "
        "cumulative-logit model includes an author random intercept and uses four chains "
        "of 4,000 retained draws. Curves show Σ k·P(score=k), with author deviations set "
        "to zero, and are restricted to each corpus’s observed publication-year range. "
        "Shading does not include model-selection uncertainty."
    )
    (OUT / "Figure_S7_caption.md").write_text(caption + "\n")


def length_report():
    allrows, diagnostics, prob = [], [], []
    for name in ADJUSTED:
        sub = pd.read_csv(LENGTH / f"{name}_quantities.csv")
        sub["adjustment"] = "Length adjusted"
        old = pd.read_csv(SOURCE / "french_books/all_model_quantities.csv")
        old = old[old.model.eq(name)].copy()
        old["adjustment"] = "Unadjusted"
        allrows.extend([old, sub])
        attempts = json.loads((LENGTH / f"{name}_attempts.json").read_text())
        diagnostics.append(
            {"model": name, "needed_refit": len(attempts) > 1, **attempts[-1]}
        )
        for adjustment, folder in [
            ("Unadjusted", SOURCE / "french_books"),
            ("Length adjusted", LENGTH),
        ]:
            draws = pd.read_csv(folder / f"{name}_shape_draws.csv")
            onset = draws["decline_0.25_year"]
            crosses = onset.notna()
            prob.append(
                {
                    "model": name,
                    "adjustment": adjustment,
                    "p_decline_1700_1799_all_draws": onset.between(1700, 1799).mean(),
                    "p_decline_1700_1799_given_crossing": onset[crosses]
                    .between(1700, 1799)
                    .mean(),
                    "p_never_crossing": (~crosses).mean(),
                }
            )
    long = pd.concat(allrows, ignore_index=True)
    long.to_csv(OUT / "Length_sensitivity_numeric_long.csv", index=False)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(OUT / "Length_sensitivity_diagnostics.csv", index=False)
    pd.DataFrame(prob).to_csv(
        OUT / "Length_sensitivity_onset_probabilities.csv", index=False
    )
    rows = []
    for name in ADJUSTED:
        for quantity, label in zip(QUANTITIES[1:], HEADERS[1:]):
            row = {"Model": model_label(name), "Quantity": label}
            for adjustment in ["Unadjusted", "Length adjusted"]:
                r = long[
                    long.model.eq(name)
                    & long.quantity.eq(quantity)
                    & long.adjustment.eq(adjustment)
                ].iloc[0]
                row[adjustment] = formatted(r, quantity)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "Length_sensitivity.csv", index=False)
    manifest = json.loads((LENGTH / "manifest.json").read_text())
    adjusted = long[long.adjustment.eq("Length adjusted")]
    flexible = adjusted[adjusted.model.str.startswith("spline")]
    onset = flexible[flexible.quantity.eq("decline_0.25_year")]
    before = long[
        long.adjustment.eq("Unadjusted")
        & long.model.str.startswith("spline")
        & long.quantity.eq("decline_0.25_year")
    ]
    probs = pd.DataFrame(prob)
    pflex = probs[
        probs.adjustment.eq("Length adjusted") & probs.model.str.startswith("spline")
    ]
    negative = adjusted[adjusted.quantity.isin(QUANTITIES[1:3])].upper_95.lt(0).all()
    eighteenth = onset["median"].between(1700, 1799).all()
    conclusion = (
        (
            "The eighteenth-century decline-threshold date survives length control across "
            "all three flexible splines. "
            if eighteenth
            else "The decline-threshold date is sensitive to model shape after length control. "
        )
        + f"For df 4–6, adjusted median dates range from {onset['median'].min():.0f} "
        f"to {onset['median'].max():.0f}, compared with {before['median'].min():.0f}–"
        f"{before['median'].max():.0f} without adjustment. "
        + (
            "The 1650→1800 and 1800→1950 contrasts remain negative, with 95% intervals "
            "entirely below zero in all four adjusted models. "
            if negative
            else "At least one adjusted contrast has a 95% interval that includes zero. "
        )
        + f"Across the adjusted splines, "
        f"{100 * pflex.p_decline_1700_1799_all_draws.min():.1f}–"
        f"{100 * pflex.p_decline_1700_1799_all_draws.max():.1f}% of all posterior draws "
        "place the 0.25-point threshold crossing in 1700–1799; "
        f"{100 * pflex.p_never_crossing.min():.1f}–{100 * pflex.p_never_crossing.max():.1f}% "
        "never cross within the observed range. These results concern the specified "
        "0.25-point threshold after each draw’s global maximum, not the exact year "
        "when the curve first begins to slope downward."
    )
    (OUT / "Length_sensitivity_summary.md").write_text(conclusion + "\n")
    report = [
        "# French books: length sensitivity",
        "",
        conclusion,
        "",
        f"All four adjusted models were refitted to the same {manifest['n']:,} records "
        f"using four chains × 4,000 retained draws, 2,000 tuning iterations, and seed 0. "
        f"The covariate is centred log₂ total segment count; predictions hold it at zero "
        f"(geometric mean {manifest['geometric_mean_segment_count']:.2f} segments). "
        "The metadata field `n_chapters` counts roughly 15,000-character text segments, "
        "not literary chapters. Author deviations are fixed at zero. Year and spline "
        "bases follow the original analysis.",
        "",
        a.md_table(table),
        "",
        "**Note.** Contrasts are later minus earlier expected scores; "
        "brackets give 95% equal-tailed intervals. Maximum dates refer to the global maximum "
        "within 1500–1996; boundary percentages combine the two endpoints. Decline-date "
        "intervals condition on crossing; never denotes no crossing within 1500–1996. "
        "The 0.25-point threshold date is not the first negative slope.",
        "",
        "## Diagnostics",
        "",
        a.md_table(
            diag[
                [
                    "model",
                    "needed_refit",
                    "divergences",
                    "max_rhat",
                    "min_ess_bulk",
                    "min_ess_tail",
                ]
            ]
        ),
        "",
        "## Probability of a 1700–1799 decline-threshold date",
        "",
        a.md_table(pd.DataFrame(prob)),
        "",
    ]
    (OUT / "Length_sensitivity.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["figure", "all", "length", "publication", "report"],
        default="figure",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "figure":
        publication_figure()
        return
    if args.stage in ("all", "publication"):
        publication_table()
        publication_figure()
    if args.stage in ("all", "length"):
        run_length()
    if args.stage == "report":
        length_report()


if __name__ == "__main__":
    main()
