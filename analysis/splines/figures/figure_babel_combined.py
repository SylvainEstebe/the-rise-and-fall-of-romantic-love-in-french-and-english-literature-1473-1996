"""Combine affiliation and signalling Babel panels using the original figure style."""

import json
from pathlib import Path

from figure_1_spline import BLUE, GREY, ORANGE, ROOT, sha256
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    output = ROOT / "results/splines/figure_data/extensions/babel_combined"
    figures = ROOT / "results/splines/figures"
    output.mkdir(parents=True, exist_ok=True)
    name = "figure_babel_affiliation_signalling_spline"
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle(
        "Romantic affiliation and signalling in Babel",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    checks, curve_rows, mean_rows = [], [], []
    for row, (folder, construct) in enumerate(
        [
            ("affiliation_babel", "Romantic affiliation"),
            ("signalling_babel", "Romantic signalling"),
        ]
    ):
        source = ROOT / "results/splines/figure_data/extensions" / folder
        manifest = json.loads((source / "manifest.json").read_text())
        curves = pd.read_csv(source / "plotted_curves.csv")
        means = pd.read_csv(source / "period_means.csv")
        selected = pd.read_csv(source / "displayed_models.csv").set_index("dataset")
        rng = np.random.default_rng(123)
        for col, (dataset, label) in enumerate(
            [
                ("french_babel", "French Babel works"),
                ("english_babel", "English Babel works"),
            ]
        ):
            ax = axes[row, col]
            path = ROOT / manifest["sources"][dataset] / "analysis_data.csv"
            assert (
                sha256(path) == manifest["source_sha256"][str(path.relative_to(ROOT))]
            )
            data = pd.read_csv(path)
            curve = curves.loc[curves.dataset.eq(dataset)].sort_values("year").copy()
            bins = means.loc[means.dataset.eq(dataset)].copy()
            visible = bins.loc[bins.displayed]
            chosen = selected.loc[dataset, "displayed_model"]
            assert curve.model.eq(chosen).all()
            np.testing.assert_array_equal(
                curve.year, np.arange(data.year.min(), data.year.max() + 1)
            )
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
            (line,) = ax.plot(
                curve.year, curve["mean"], color=ORANGE, linewidth=2.5, zorder=3
            )
            np.testing.assert_array_equal(line.get_ydata(), curve["mean"])
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
            ax.set_title(f"{label} (n = {len(data):,})", pad=9)
            ax.text(
                -0.02,
                1.04,
                "abcd"[row * 2 + col],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(color="#EEEEEE", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set(
                xlim=(1450, 2025),
                ylim=(-0.45, 6.45),
                xticks=np.arange(1500, 2001, 100),
                yticks=np.arange(7),
            )
            if row == 1:
                ax.set_xlabel("Year")
            checks.append(
                dict(
                    construct=manifest["construct"],
                    dataset=dataset,
                    model=chosen,
                    n=len(data),
                    source=str(source.relative_to(ROOT)),
                    curve_sha256=sha256(source / "plotted_curves.csv"),
                    means_sha256=sha256(source / "period_means.csv"),
                )
            )
            curve.insert(0, "construct", manifest["construct"])
            bins.insert(0, "construct", manifest["construct"])
            curve_rows.append(curve)
            mean_rows.append(bins)
    summary = axes[0, 0].errorbar(
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
            label="Knowledge-based ratings",
        ),
        Line2D([], [], color=ORANGE, lw=2.5, label="Ordinal spline trend"),
        Patch(facecolor=ORANGE, alpha=0.14, label="Pointwise 95% credible interval"),
        summary,
    ]
    fig.supylabel("Knowledge-based rating (0–6)", x=0.025, fontsize=12)
    fig.subplots_adjust(
        left=0.085, right=0.99, top=0.84, bottom=0.16, hspace=0.48, wspace=0.08
    )
    for row, label in enumerate(["Romantic affiliation", "Romantic signalling"]):
        fig.text(
            0.085,
            axes[row, 0].get_position().y1 + 0.065,
            label,
            fontsize=12,
            fontweight="bold",
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
    for ext in ["png", "pdf"]:
        fig.savefig(
            figures / f"{name}.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
        )
    plt.close(fig)
    caption_source = Path(__file__).with_name("babel_caption.md")
    caption = (
        caption_source.read_text()
        .replace("Muted grey-green", "Grey")
        .replace("Forest-green", "Blue")
        .replace("Raspberry curves", "Orange curves")
        .replace("Lilac hatched bands", "Pale orange bands")
    )
    (figures / f"{name}_caption.md").write_text(caption)
    (output / "caption.md").write_text(caption)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                style="original white/blue/orange",
                panels=checks,
                refitted=False,
                png_dpi=300,
            ),
            indent=2,
        )
    )
    pd.concat(curve_rows, ignore_index=True).to_csv(
        output / "plotted_curves.csv", index=False
    )
    pd.concat(mean_rows, ignore_index=True).to_csv(
        output / "period_means.csv", index=False
    )
    print(figures / f"{name}.png")


if __name__ == "__main__":
    main()
