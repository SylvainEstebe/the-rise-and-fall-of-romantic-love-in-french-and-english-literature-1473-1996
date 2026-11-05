"""Reproduce manuscript S1's 25-year histograms without the reference line."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = [
    ("fr_novels_text", "French novels"),
    ("uk_novels_text", "English novels"),
    ("theatre_fr_text", "French plays"),
    ("theatre_uk_text", "English plays"),
]


def main():
    from four_corpora import load_corpus

    names = dict(
        zip(
            [key for key, _ in PANELS],
            ["roland_fr", "novels_uk", "theatre_fr", "theatre_uk"],
        )
    )
    years = {key: load_corpus(names[key]).year.to_numpy() for key, _ in PANELS}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.24, 6.39), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.078, right=0.973, bottom=0.115, top=0.895, wspace=0.13, hspace=0.23
    )
    fig.suptitle(
        "Temporal distribution of works across the four corpora",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    edges = np.arange(1450, 2025, 25)
    for ax, (key, title), letter in zip(axes.flat, PANELS, "abcd"):
        counts, _ = np.histogram(years[key], bins=edges)
        assert counts.sum() == len(years[key])
        ax.bar(
            edges[:-1] + 12.5,
            counts,
            width=23,
            color="#3592c2",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax.set(xlim=(1450, 2000), ylim=(0, 175), title=title)
        ax.set_xticks(np.arange(1500, 2001, 100))
        ax.set_yticks(np.arange(0, 175, 25))
        ax.set_axisbelow(True)
        ax.grid(color="#eeeeee", linewidth=0.5, alpha=0.8)
        ax.text(
            -0.02,
            1.04,
            letter,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10.5,
        )
        print(f"{key}: {counts.sum()} works")
    fig.supxlabel("Year", x=0.49, y=0.018, fontsize=11)
    fig.supylabel("Number of works per 25-year period", x=0.01, fontsize=11)
    out = ROOT / "results/primary/figures/figure_s1_histograms"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix("." + ext), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
