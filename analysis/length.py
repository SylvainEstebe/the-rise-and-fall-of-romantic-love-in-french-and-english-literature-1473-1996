"""French source-length sensitivity with the original Bayesian specification."""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/romantic_length_mpl")
os.environ.setdefault("PYTENSOR_FLAGS", "compiledir=/tmp/romantic_length_pytensor")
import json, hashlib, gc, warnings
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.style.core  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import runtime  # noqa: F401
import bambi as bmb
import arviz as az
import pymc as pm
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/length"
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = ROOT


def load():
    src = PRIMARY / "data/annotations/main/roland_fr"
    scores = pd.read_json(src / "annotations_gpt.json", dtype={"id": str}).drop(
        columns="date"
    )
    meta = pd.read_csv(src / "metadata_roland_annotations.csv", dtype={"id": str})
    assert scores.id.is_unique and meta.id.is_unique
    d = scores.merge(meta[["id", "author", "date"]], on="id", validate="one_to_one")
    d["year"] = pd.to_numeric(d.date)
    d["score"] = pd.to_numeric(d.ING_38_Role)
    d = d.dropna(subset=["year", "score"]).copy()
    d["author"] = d.author.replace(["Auteur inconnu", "Unknown Author"], np.nan).fillna(
        "unknown_" + d.id
    )
    frozen = pd.read_csv(ROOT / "data/metadata/roland_length.csv")
    d = d.merge(
        frozen[["id", "year", "ING_38", "n_chapters", "char_count", "chapter_idx"]],
        on="id",
        validate="one_to_one",
        suffixes=("", "_frozen"),
    )
    assert len(d) == 1879 and d.author.nunique() == 1264
    assert (d.year == d.year_frozen).all() and (d.score == d.ING_38).all()
    d = d[
        ["id", "author", "year", "score", "n_chapters", "char_count", "chapter_idx"]
    ].copy()
    d.to_csv(OUT / "linked_data.csv", index=False)
    paths = [
        src / "annotations_gpt.json",
        src / "metadata_roland_annotations.csv",
        ROOT / "analysis/four_corpora.py",
        ROOT / "data/metadata/roland_length.csv",
    ]
    manifest = {
        "inputs": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "n": len(d),
        "authors": d.author.nunique(),
        "draws_per_chain": 4000,
        "tune_per_chain": 2000,
        "chains": 4,
        "seed": 0,
        "versions": {
            "bambi": bmb.__version__,
            "pymc": pm.__version__,
            "arviz": az.__version__,
        },
        "restricted_length": [12, 32],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


def expected(post, df, years, adjusted):
    # Thresholds and linear predictor reproduce Bambi p at zero author deviation.
    flat = post.stack(sample=("chain", "draw"))
    thresholds = flat["threshold"].transpose("sample", ...).values
    x = (np.asarray(years) - df.year.mean()) / 100
    eta = flat["centered_year"].values[:, None] * x[None, :]
    if "I(centered_year ** 2)" in flat:
        eta += flat["I(centered_year ** 2)"].values[:, None] * x[None, :] ** 2
    # centered_log2_length = 0, the same geometric-mean source length at all years.
    return expit(eta[:, :, None] - thresholds[:, None, :]).sum(axis=2)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Resample instead of loading saved posterior draws",
    )
    args = parser.parse_args()
    d = load()
    diags = []
    contrasts = []
    curves = []
    comparisons = []
    coeff = []
    for variant in ["original", "length_adjusted", "restricted_length"]:
        z = (
            d.loc[d.n_chapters.between(12, 32)].copy()
            if variant == "restricted_length"
            else d.copy()
        )
        z["centered_year"] = (z.year - z.year.mean()) / 100
        z["centered_log2_length"] = np.log2(z.n_chapters) - np.log2(z.n_chapters).mean()
        z["score_categories"] = pd.Categorical(
            z.score.astype(int), categories=range(7), ordered=True
        )
        z = z[
            [
                "id",
                "author",
                "year",
                "score",
                "n_chapters",
                "centered_year",
                "centered_log2_length",
                "score_categories",
            ]
        ]
        loos = {}
        for shape in ["linear", "quadratic"]:
            name = f"{variant}_{shape}"
            formula = (
                "score_categories ~ centered_year"
                + (" + I(centered_year**2)" if shape == "quadratic" else "")
                + (" + centered_log2_length" if variant == "length_adjusted" else "")
                + " + (1|author)"
            )
            m = bmb.Model(formula, z, family="cumulative")
            (OUT / f"{name}_model.txt").write_text(str(m))
            path = OUT / f"{name}_posterior.nc"
            if path.exists() and (OUT / f"{name}_loo.json").exists() and not args.refit:
                idata = az.from_netcdf(path)
                lj = json.loads((OUT / f"{name}_loo.json").read_text())
                loos[shape] = lj
            else:
                print("FITTING", name, len(z), flush=True)
                idata = m.fit(
                    draws=4000,
                    tune=2000,
                    chains=4,
                    cores=4,
                    random_seed=0,
                    progressbar=False,
                    idata_kwargs={"log_likelihood": True},
                )
                summary = az.summary(idata, round_to="none")
                if (
                    int(idata.sample_stats.diverging.sum()) > 0
                    or summary.r_hat.max() > 1.01
                ):
                    print(
                        "REPAIR: increase tuning and target acceptance",
                        name,
                        flush=True,
                    )
                    del idata
                    gc.collect()
                    idata = m.fit(
                        draws=4000,
                        tune=4000,
                        chains=4,
                        cores=4,
                        random_seed=0,
                        target_accept=0.95,
                        progressbar=False,
                        idata_kwargs={"log_likelihood": True},
                    )
                with warnings.catch_warnings(record=True) as w:
                    loo = az.loo(idata, pointwise=True)
                lj = {
                    "elpd_loo": float(loo.elpd_loo),
                    "se": float(loo.se),
                    "p_loo": float(loo.p_loo),
                    "loo_i": loo.loo_i.values.tolist(),
                    "pareto_k_max": float(loo.pareto_k.max()),
                    "pareto_k_above_07": int((loo.pareto_k > 0.7).sum()),
                    "warnings": [str(a.message) for a in w],
                }
                (OUT / f"{name}_loo.json").write_text(json.dumps(lj, indent=2))
                loos[shape] = lj
                del idata.log_likelihood
                if "p" in idata.posterior:
                    idata.posterior = idata.posterior.drop_vars("p")
                idata.to_netcdf(path)
            summary = az.summary(idata, round_to="none")
            summary.to_csv(OUT / f"{name}_parameters.csv")
            diag = {
                "variant": variant,
                "shape": shape,
                "n": len(z),
                "authors": z.author.nunique(),
                "divergences": int(idata.sample_stats.diverging.sum()),
                "max_rhat": float(summary.r_hat.max()),
                "min_ess_bulk": float(summary.ess_bulk.min()),
                "min_ess_tail": float(summary.ess_tail.min()),
                "pareto_k_max": lj["pareto_k_max"],
                "pareto_k_above_07": lj["pareto_k_above_07"],
            }
            diags.append(diag)
            for param in [
                "centered_year",
                "I(centered_year ** 2)",
                "centered_log2_length",
                "1|author_sigma",
            ]:
                if param in summary.index:
                    coeff.append(
                        {
                            "variant": variant,
                            "shape": shape,
                            "parameter": param,
                            **summary.loc[param].to_dict(),
                        }
                    )
            if shape == "quadratic":
                years = np.unique(np.r_[np.linspace(1500, 1996, 180), 1800, 1900, 1996])
                sims = expected(idata.posterior, z, years, variant == "length_adjusted")
                # Cross-check custom expected-score calculation against Bambi predictions.
                test = pd.DataFrame(
                    {
                        "centered_year": (np.array([1800, 1900, 1996]) - z.year.mean())
                        / 100,
                        "centered_log2_length": 0.0,
                    }
                )
                bp = m.predict(
                    idata, data=test, include_group_specific=False, inplace=False
                ).posterior["p"]
                be = (
                    (bp * np.arange(7))
                    .sum(bp.dims[-1])
                    .stack(sample=("chain", "draw"))
                    .transpose("sample", ...)
                    .values
                )
                np.testing.assert_allclose(
                    be, sims[:, np.searchsorted(years, [1800, 1900, 1996])], atol=1e-10
                )
                q = np.quantile(sims, [0.025, 0.975], axis=0)
                curves.extend(
                    {
                        "variant": variant,
                        "year": yr,
                        "mean": mean,
                        "low": low,
                        "high": high,
                    }
                    for yr, mean, low, high in zip(years, sims.mean(0), *q)
                )
                for start, end in [(1800, 1900), (1900, 1996), (1800, 1996)]:
                    diff = (
                        sims[:, np.searchsorted(years, end)]
                        - sims[:, np.searchsorted(years, start)]
                    )
                    contrasts.append(
                        {
                            "variant": variant,
                            "start": start,
                            "end": end,
                            "n": len(z),
                            "mean_change": float(diff.mean()),
                            "low": float(np.quantile(diff, 0.025)),
                            "high": float(np.quantile(diff, 0.975)),
                            "prob_decline": float((diff < 0).mean()),
                        }
                    )
                print("RESULT", name, contrasts[-3:], diag, flush=True)
            del idata, m
            gc.collect()
        ld = np.array(loos["quadratic"]["loo_i"]) - np.array(loos["linear"]["loo_i"])
        comparisons.append(
            {
                "variant": variant,
                "delta_elpd_quad_minus_linear": float(ld.sum()),
                "se": float(np.sqrt(len(ld) * ld.var(ddof=1))),
            }
        )
        for name, data in [
            ("diagnostics", diags),
            ("contrasts", contrasts),
            ("curves", curves),
            ("comparison", comparisons),
            ("coefficients", coeff),
        ]:
            pd.DataFrame(data).to_csv(OUT / f"{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for variant, color in [
        ("original", "#444444"),
        ("length_adjusted", "#c26927"),
        ("restricted_length", "#168794"),
    ]:
        q = pd.DataFrame(curves).query("variant == @variant")
        ax.plot(q.year, q["mean"], label=variant.replace("_", " "), color=color)
        ax.fill_between(q.year, q.low, q.high, color=color, alpha=0.13)
    ax.set(
        xlabel="Publication year",
        ylabel="Expected romantic affiliation score",
        ylim=(0, 6),
        title="French books: Bayesian sensitivity to source-text length",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"length_sensitivity.{ext}", dpi=180)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    run()
