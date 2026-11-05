"""Page-sized English panels for the Google Doc, from verified result tables."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/romantic_length_mpl")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from descriptive import LABELS, OUT

DEST = OUT / "doc_assets"
DEST.mkdir(exist_ok=True)
periods = pd.read_csv(OUT / "observed_periods.csv")
curves = pd.read_csv(OUT / "late_models/model_curves.csv")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)
groups = {
    "a": ["roland_fr", "theatre_fr", "babel_fr"],
    "b": ["novels_uk", "theatre_uk", "babel_uk"],
}
for family, number in [("observed", 4), ("threshold", 5), ("models", 6)]:
    for part, keys in groups.items():
        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        for row, key in enumerate(keys):
            threshold = 3 if family == "threshold" else 1
            for col, (metric, color) in enumerate([("p", "#20689a"), ("m", "#bf6724")]):
                ax = axes[row, col]
                if family == "models":
                    d = pd.read_csv(OUT / f"{key}_data.csv")
                    lo = 1700 if key.startswith("theatre") else 1800
                    d = d[d.year.between(lo, min(1996, int(d.year.max())))].copy()
                    d["period"] = d.year // 20 * 20
                    obs = []
                    for _, g in d.groupby("period"):
                        pos = g[g.score > 0]
                        obs.append(
                            {
                                "year": g.year.mean(),
                                "n": len(g),
                                "n_positive": len(pos),
                                "p": len(pos) / len(g),
                                "m": pos.score.mean(),
                            }
                        )
                    obs = pd.DataFrame(obs)
                    stage = "occurrence" if metric == "p" else "positive_importance"
                    q = curves.query(
                        'corpus==@key and variant=="original" and stage==@stage'
                    )
                    ax.plot(q.year, q["mean"], color=color, lw=2)
                    ax.fill_between(q.year, q.low, q.high, color=color, alpha=0.16)
                    v = (
                        obs[obs.n >= 10]
                        if metric == "p"
                        else obs[(obs.n >= 10) & (obs.n_positive >= 10)]
                    )
                    ax.scatter(
                        v.year, v[metric], s=22, c="#777777", alpha=0.75, zorder=3
                    )
                    if key == "roland_fr":
                        a = curves.query(
                            'corpus==@key and variant=="length_adjusted" and stage==@stage'
                        )
                        ax.plot(
                            a.year,
                            a["mean"],
                            ls="--",
                            color="#222222",
                            label="Length adjusted",
                        )
                        ax.legend(frameon=False, loc="best")
                    ax.set_xlim(d.year.min(), d.year.max())
                else:
                    q = periods.query(
                        'corpus==@key and variant=="all" and threshold==@threshold'
                    ).sort_values("start")
                    v = q.copy()
                    mask = v.n < 10
                    if metric == "m":
                        mask |= v.n_positive < 10
                    v.loc[mask, [metric, metric + "_low", metric + "_high"]] = np.nan
                    grid = pd.DataFrame(
                        {"start": np.arange(q.start.min(), q.start.max() + 1, 20)}
                    )
                    v = grid.merge(v, on="start", how="left")
                    v["plot_year"] = v.year.fillna(v.start + 10)
                    ax.plot(v.plot_year, v[metric], "-o", ms=4, color=color, lw=1.5)
                    ax.fill_between(
                        v.plot_year,
                        v[metric + "_low"],
                        v[metric + "_high"],
                        color=color,
                        alpha=0.16,
                    )
                ax.set_ylim((0, 1.03) if metric == "p" else (threshold - 0.1, 6.1))
                ax.set_title(LABELS[key], pad=8)
                ax.grid(alpha=0.18)
                ax.spines[["top", "right"]].set_visible(False)
                rule = "> 0" if threshold == 1 else "≥ 3"
                ax.set_ylabel(
                    f"Proportion scoring {rule}"
                    if metric == "p"
                    else f"Mean score given score {rule}"
                )
                ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
                if metric == "p":
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_xlabel("Publication year")
        fig.subplots_adjust(
            left=0.09, right=0.99, top=0.96, bottom=0.07, wspace=0.30, hspace=0.72
        )
        for ext in ["png", "pdf"]:
            fig.savefig(
                DEST / f"figure_s{number}{part}.{ext}", dpi=220, facecolor="white"
            )
        plt.close(fig)
print(DEST)
