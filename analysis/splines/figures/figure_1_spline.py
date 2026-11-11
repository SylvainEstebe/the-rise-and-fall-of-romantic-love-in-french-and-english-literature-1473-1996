#!/usr/bin/env python3
"""Replace the four-corpus Figure 1 trend with one fitted cubic spline per panel.

Run from the repository root: python script/figures/figure_1_spline.py
Uses the final 4-chain × 4,000-draw fits; never resamples. Selects the highest
LOO ELPD among the genuinely flexible df=4,5,6 candidates in each corpus.
The original 25-year descriptive summaries, filtering and 2×2 layout are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/love_spline_mpl")
import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/splines/affiliation"
OUTPUT = ROOT / "results/splines/figure_data/main"
FIGURES = ROOT / "results/splines/figures"
CORPORA = {
    "french_books": "French books",
    "english_books": "English books",
    "french_theatre": "French plays",
    "english_theatre": "English plays",
}
CANDIDATES = ["spline_4", "spline_5", "spline_6"]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#8A8A8A"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(
    *,
    corpora=None,
    sources=None,
    output=OUTPUT,
    figure_dir=FIGURES,
    filename="figure_1_spline",
    construct="Romantic affiliation",
    code="ING_38",
    mode="text",
    verify_draws=False,
):
    """Render either four text corpora or two Babel corpora using the same style."""
    corpora = CORPORA if corpora is None else corpora
    sources = {} if sources is None else sources
    assert len(corpora) in (2, 4)
    title = (
        f"{construct} in French and English literature"
        if mode == "text"
        else f"{construct} in Babel: knowledge-based ratings"
    )
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = len(corpora) // 2
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(10, 7 if rows == 2 else 4.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    rng = np.random.default_rng(123)
    selected, curve_rows, mean_rows, input_hashes = [], [], [], {}
    for i, ((dataset, panel_title), ax) in enumerate(zip(corpora.items(), axes.flat)):
        source = sources.get(dataset, SOURCE / dataset)
        data = pd.read_csv(source / "analysis_data.csv")
        loo = pd.read_csv(source / "loo_full.csv").set_index("model")
        chosen = loo.loc[CANDIDATES, "elpd_loo"].idxmax()
        selection = loo.loc[chosen].to_dict()
        selection.update(
            dataset=dataset,
            displayed_model=chosen,
            selection_scope="highest ELPD among spline df 4, 5, 6",
        )
        selected.append(selection)
        curves = pd.read_csv(source / "all_curves.csv")
        curve = curves.loc[curves.model.eq(chosen)].sort_values("year").copy()
        assert curve.year.is_unique
        np.testing.assert_array_equal(
            curve.year, np.arange(data.year.min(), data.year.max() + 1)
        )
        # Optional audit against the separately distributed predictive draws.
        if verify_draws:
            # Independently check the plotted mean and band against all 16,000 draws.
            with np.load(source / f"{chosen}_expected.npz") as saved:
                expected, years = saved["expected"], saved["years"]
                assert expected.shape == (16000, len(curve))
                np.testing.assert_array_equal(years, curve.year)
                assert (
                    np.isfinite(expected).all()
                    and (expected >= 0).all()
                    and (expected <= 6).all()
                )
                np.testing.assert_allclose(curve["mean"], expected.mean(0), atol=1e-10)
                np.testing.assert_allclose(
                    curve.lower_95, np.quantile(expected, 0.025, axis=0), atol=1e-10
                )
                np.testing.assert_allclose(
                    curve.upper_95, np.quantile(expected, 0.975, axis=0), atol=1e-10
                )
        curve.insert(0, "dataset", dataset)
        curve_rows.append(curve)
        data["period"] = 25 * np.floor(data.year / 25)
        means = (
            data.groupby("period")
            .agg(
                year=("year", "mean"),
                mean_score=("score", "mean"),
                standard_error=("score", "sem"),
                number_of_works=("score", "size"),
            )
            .reset_index()
        )
        means["lower"] = means.mean_score - 1.96 * means.standard_error
        means["upper"] = means.mean_score + 1.96 * means.standard_error
        means["interval_width"] = means.upper - means.lower
        means["displayed"] = (means.number_of_works >= 10) & (means.interval_width <= 2)
        means.insert(0, "dataset", dataset)
        mean_rows.append(means)
        visible = means.loc[means.displayed]
        ax.scatter(
            data.year + rng.uniform(-1.5, 1.5, len(data)),
            data.score + rng.uniform(-0.18, 0.18, len(data)),
            c=GREY,
            s=5,
            alpha=0.13,
            linewidths=0,
            rasterized=True,
        )
        ax.fill_between(
            curve.year,
            curve.lower_95,
            curve.upper_95,
            color=ORANGE,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(curve.year, curve["mean"], color=ORANGE, linewidth=2.5, zorder=3)
        ax.errorbar(
            visible.year,
            visible.mean_score,
            yerr=1.96 * visible.standard_error,
            fmt="o",
            color=BLUE,
            ms=4,
            mec="white",
            mew=0.5,
            elinewidth=1,
            capsize=2,
            zorder=4,
        )
        ax.set_title(f"{panel_title} (n = {len(data):,})", pad=9)
        ax.text(
            -0.02,
            1.04,
            "abcd"[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(color="#EEEEEE", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set(
            xlim=(1450, 2025 if mode == "babel" else 2000),
            ylim=(-0.45, 6.45),
            xticks=np.arange(1500, 2001, 100),
            yticks=np.arange(7),
        )
        if i >= len(corpora) - 2:
            ax.set_xlabel("Year")
        for source_filename in [
            "analysis_data.csv",
            "loo_full.csv",
            "all_curves.csv",
            f"{chosen}_attempts.json",
        ]:
            p = source / source_filename
            input_hashes[str(p.relative_to(ROOT))] = sha256(p)
    # A single common key preserves the uncluttered original layout.
    summary_handle = axes[0, 0].errorbar(
        [],
        [],
        yerr=[],
        fmt="o",
        color=BLUE,
        ms=4,
        capsize=2,
        elinewidth=1,
        label="25-year mean ± 1.96 SEM",
    )
    handles = [
        Line2D(
            [],
            [],
            color=GREY,
            marker="o",
            markersize=3,
            ls="",
            alpha=0.4,
            label="Individual annotations"
            if mode == "text"
            else "Knowledge-based ratings",
        ),
        Line2D([], [], color=ORANGE, lw=2.5, label="Ordinal spline trend"),
        Patch(facecolor=ORANGE, alpha=0.14, label="Pointwise 95% credible interval"),
        summary_handle,
    ]
    fig.supylabel(
        "Annotation score (0–6)" if mode == "text" else "Knowledge-based rating (0–6)",
        x=0.025,
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.89 if rows == 2 else 0.80,
        bottom=0.18 if rows == 2 else 0.27,
        hspace=0.24,
        wspace=0.08,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.015),
        ncol=2,
        frameon=False,
        fontsize=9,
        columnspacing=2.4,
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
        )
    plt.close(fig)
    selection = pd.DataFrame(selected)
    selection.to_csv(output / "displayed_models.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(
        output / "plotted_curves.csv", index=False
    )
    pd.concat(mean_rows, ignore_index=True).to_csv(
        output / "period_means.csv", index=False
    )
    choices = "; ".join(
        f"{corpora[r.dataset]}, df = {r.displayed_model[-1]}"
        for r in selection.itertuples()
    )
    opening = (
        f"**Figure 1. {title} ({code}).** "
        if filename == "figure_1_spline"
        else f"**{title} ({code}).** "
    )
    panels = (
        "Panels show French books (a), English books (b), French plays (c), "
        "and English plays (d). "
        if mode == "text"
        else "Panels show French Babel works (a) and English Babel works (b). "
        "These are knowledge-based ratings elicited from bibliographic information, "
        "not direct text annotations. French works were rated by GPT and the processed "
        "English subset by Claude; cross-language differences also reflect model "
        "and corpus-coverage differences. "
    )
    caption = (
        opening
        + panels
        + "Grey points are individual ratings jittered for visibility. Blue points show means "
        "in calendar-aligned 25-year bins, positioned at the mean publication year, with "
        "error bars of ±1.96 standard errors of the mean. Bins are displayed only when they "
        "contain at least 10 observations and the interval is no wider than 2 score points. "
        "The orange curve shows the posterior mean expected score from a cumulative-logit "
        "model with a cubic B-spline in publication year and an author random intercept. "
        "One spline per corpus is selected by highest PSIS-LOO ELPD among df = 4, 5, and 6 "
        "(" + choices + "). Flagged LOO observations are corrected by exact refitting. "
        "Shading shows the pointwise 95% equal-tailed credible interval conditional on "
        "the displayed specification. Predictions use four chains × 4,000 retained draws, "
        "fix author deviations to zero, and are restricted to the observed publication-year "
        "range. Expected scores are Σ k·P(score=k)."
    )
    if filename == "figure_1_spline":
        caption += (
            " Alternative specifications and their comparisons are reported in "
            "Supplementary Figure S7 and Table S10. Timing estimates in the text "
            "summarize dates calculated within each posterior draw and need not "
            "coincide with the maximum of the plotted posterior mean curve."
        )
        (output / "Figure_1_caption.md").write_text(caption + "\n")
    (output / f"{filename}_caption.md").write_text(caption + "\n")
    (figure_dir / f"{filename}_caption.md").write_text(caption + "\n")
    manifest = {
        "construct": code,
        "title": title,
        "mode": mode,
        "selection": choices,
        "candidate_df": [4, 5, 6],
        "sources": {
            k: str(sources.get(k, SOURCE / k).relative_to(ROOT)) for k in corpora
        },
        "source_sha256": input_hashes,
        "original_style_reference": "results/primary/figures/figure_ordinal.png",
        "posterior_draws_verified": verify_draws,
        "layout": f"{rows} × 2, common axes, grey ratings, blue 25-year means, one orange curve",
        "preserved_bin_rules": "25 years; n >= 10; interval width <= 2",
        "posterior": "final unadjusted fits, 4 chains × 4000 retained draws",
        "author_deviations": 0,
        "refitted": False,
        "verification": (
            "all selected curves and bands checked against 16000 expected-score draws"
            if verify_draws
            else "redrawn from frozen curve summaries; use --verify-draws for a posterior audit"
        ),
        "s7": "unchanged; S7 uses 20-year bins, whereas Figure 1 retains its original 25-year bins",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        selection[
            ["dataset", "displayed_model", "delta_elpd_loss_vs_best", "delta_elpd_se"]
        ].to_string(index=False)
    )
    print(figure_dir / f"{filename}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-draws",
        action="store_true",
        help="Check curves against all 16,000 saved draws; requires the release draw archives",
    )
    main(verify_draws=parser.parse_args().verify_draws)
