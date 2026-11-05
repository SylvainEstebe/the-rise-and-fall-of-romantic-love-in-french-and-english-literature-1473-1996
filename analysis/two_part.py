"""Independent Bayesian occurrence and positive-score ordinal models."""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/romantic_length_mpl")
os.environ.setdefault("PYTENSOR_FLAGS", "compiledir=/tmp/romantic_length_pytensor")
import gc
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.style.core  # noqa: F401
import runtime  # noqa: F401
import bambi as bmb, arviz as az
from scipy.special import expit
from descriptive import LABELS, OUT as SOURCE

OUT = SOURCE / "late_models"
OUT.mkdir(exist_ok=True)


def summarize(v):
    return {
        "mean": float(np.mean(v)),
        "low": float(np.quantile(v, 0.025)),
        "high": float(np.quantile(v, 0.975)),
        "prob_negative": float(np.mean(v < 0)),
    }


def predictor(post, years, center):
    p = post.stack(sample=("chain", "draw"))
    x = (years - center) / 100
    eta = p.x.values[:, None] * x
    if "I(x ** 2)" in p:
        eta += p["I(x ** 2)"].values[:, None] * x * x
    if "Intercept" in p:
        eta += p.Intercept.values[:, None]
    return p, eta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Resample instead of loading saved posterior draws",
    )
    args = parser.parse_args()
    curves = []
    contrasts = []
    diagnostics = []
    allcoef = []
    for ci, key in enumerate(LABELS):
        full = pd.read_csv(SOURCE / f"{key}_data.csv")
        for variant in (
            ["original", "length_adjusted"] if key == "roland_fr" else ["original"]
        ):
            start_year = 1700 if key.startswith("theatre") else 1800
            d = full[
                full.year.between(start_year, min(1996, int(full.year.max())))
            ].copy()
            center = d.year.mean()
            d["x"] = (d.year - center) / 100
            if variant == "length_adjusted":
                d["length"] = np.log2(d.n_chapters) - np.log2(d.n_chapters).mean()
            shape = "linear"
            rhs = (
                "x"
                + (" + I(x**2)" if shape == "quadratic" else "")
                + (" + length" if variant == "length_adjusted" else "")
                + " + (1|author)"
            )
            years = np.unique(
                np.r_[
                    np.linspace(d.year.min(), d.year.max(), 160),
                    [
                        y
                        for y in [1600, 1700, 1800, 1900, 1949, 1996]
                        if d.year.min() <= y <= d.year.max()
                    ],
                ]
            )
            sims = {}
            for si, stage in enumerate(["occurrence", "positive_importance"]):
                name = f"{key}_{variant}_{stage}"
                z = d if stage == "occurrence" else d[d.score > 0].copy()
                z = z.copy()
                z["present"] = (z.score > 0).astype(int)
                z["positive_score"] = pd.Categorical(
                    z.score, categories=range(1, 7), ordered=True
                )
                outcome = "present" if stage == "occurrence" else "positive_score"
                m = bmb.Model(
                    f"{outcome} ~ {rhs}",
                    z,
                    family="bernoulli" if stage == "occurrence" else "cumulative",
                )
                (OUT / f"{name}_model.txt").write_text(str(m))
                path = OUT / f"{name}_posterior.nc"
                seed = 42 + ci * 10 + si + (100 if variant == "length_adjusted" else 0)
                if path.exists() and not args.refit:
                    idata = az.from_netcdf(path)
                else:
                    print("FITTING", name, "n=", len(z), flush=True)
                    idata = m.fit(
                        draws=4000,
                        tune=2000,
                        chains=4,
                        cores=4,
                        random_seed=seed,
                        target_accept=0.9,
                        progressbar=False,
                    )
                    if "p" in idata.posterior:
                        idata.posterior = idata.posterior.drop_vars("p")
                    s = az.summary(idata, round_to="none")
                    if (
                        int(idata.sample_stats.diverging.sum()) > 0
                        or s.r_hat.max() > 1.01
                    ):
                        print("REPAIR", name, flush=True)
                        del idata
                        gc.collect()
                        idata = m.fit(
                            draws=4000,
                            tune=4000,
                            chains=4,
                            cores=4,
                            random_seed=seed,
                            target_accept=0.97,
                            progressbar=False,
                        )
                        if "p" in idata.posterior:
                            idata.posterior = idata.posterior.drop_vars("p")
                    idata.to_netcdf(path)
                s = az.summary(idata, round_to="none")
                if int(idata.sample_stats.diverging.sum()) > 0 or s.r_hat.max() > 1.01:
                    print("HIGH_ACCURACY_REPAIR", name, flush=True)
                    del idata
                    gc.collect()
                    idata = m.fit(
                        draws=4000,
                        tune=6000,
                        chains=4,
                        cores=4,
                        random_seed=seed,
                        target_accept=0.995,
                        max_treedepth=14,
                        progressbar=False,
                    )
                    if "p" in idata.posterior:
                        idata.posterior = idata.posterior.drop_vars("p")
                    idata.to_netcdf(path)
                    s = az.summary(idata, round_to="none")
                s.to_csv(OUT / f"{name}_parameters.csv")
                diag = {
                    "corpus": key,
                    "variant": variant,
                    "stage": stage,
                    "shape": shape,
                    "n": len(z),
                    "authors": z.author.nunique(),
                    "divergences": int(idata.sample_stats.diverging.sum()),
                    "max_rhat": float(s.r_hat.max()),
                    "min_ess_bulk": float(s.ess_bulk.min()),
                    "min_ess_tail": float(s.ess_tail.min()),
                }
                diagnostics.append(diag)
                for param in ["x", "I(x ** 2)", "length", "1|author_sigma"]:
                    if param in s.index:
                        allcoef.append(
                            {
                                "corpus": key,
                                "variant": variant,
                                "stage": stage,
                                "parameter": param,
                                **s.loc[param].to_dict(),
                            }
                        )
                post, eta = predictor(idata.posterior, years, center)
                if stage == "occurrence":
                    v = expit(eta)
                else:
                    th = post.threshold.transpose("sample", ...).values
                    assert th.shape[1] == 5
                    v = 1 + expit(eta[:, :, None] - th[:, None, :]).sum(2)
                # Verify formula/sign and category mapping against Bambi predictions.
                ix = np.array([0, len(years) // 2, len(years) - 1])
                test = pd.DataFrame({"x": (years[ix] - center) / 100, "length": 0.0})
                pred = m.predict(
                    idata, data=test, include_group_specific=False, inplace=False
                ).posterior["p"]
                if stage != "occurrence":
                    pred = (pred * np.arange(1, 7)).sum(pred.dims[-1])
                bp = (
                    pred.stack(sample=("chain", "draw")).transpose("sample", ...).values
                )
                np.testing.assert_allclose(bp, v[:, ix], atol=1e-10)
                # Independent permutations make the independence of stage posteriors explicit.
                sims[stage] = v[np.random.default_rng(seed + 1000).permutation(len(v))]
                q = np.quantile(v, [0.025, 0.975], axis=0)
                curves.extend(
                    {
                        "corpus": key,
                        "variant": variant,
                        "stage": stage,
                        "year": yr,
                        "mean": av,
                        "low": lo,
                        "high": hi,
                    }
                    for yr, av, lo, hi in zip(years, v.mean(0), *q)
                )
                print("FINISHED", name, diag, flush=True)
                del idata, m
                gc.collect()
            pairs = (
                [(1600, 1700), (1700, 1800)]
                if key == "theatre_uk"
                else [(1700, 1800), (1800, 1899)]
                if key == "theatre_fr"
                else [(1800, 1900), (1900, min(1996, int(d.year.max())))]
            )
            # Add exact contrasts at observed endpoints not initially included.
            for lo, hi in pairs:
                if lo < years.min() or hi > years.max():
                    continue
                # theatre endpoints 1899 / 1800 are already observed min/max included by grid.
                i = np.where(np.isclose(years, lo))[0]
                j = np.where(np.isclose(years, hi))[0]
                assert len(i) == len(j) == 1, (key, lo, hi)
                i, j = i[0], j[0]
                p0, p1 = sims["occurrence"][:, i], sims["occurrence"][:, j]
                m0, m1 = (
                    sims["positive_importance"][:, i],
                    sims["positive_importance"][:, j],
                )
                A = (p1 - p0) * (m1 + m0) / 2
                B = (m1 - m0) * (p1 + p0) / 2
                total = p1 * m1 - p0 * m0
                np.testing.assert_allclose(A + B, total, atol=1e-12)
                row = {"corpus": key, "variant": variant, "start": lo, "end": hi}
                for name, v in [
                    ("prevalence_change", p1 - p0),
                    ("importance_change", m1 - m0),
                    ("prevalence_contribution", A),
                    ("importance_contribution", B),
                    ("total_change", total),
                    ("prevalence_start", p0),
                    ("prevalence_end", p1),
                    ("importance_start", m0),
                    ("importance_end", m1),
                ]:
                    row.update(
                        {name + "_" + k: value for k, value in summarize(v).items()}
                    )
                contrasts.append(row)
            for name, rows in [
                ("model_curves", curves),
                ("model_contrasts", contrasts),
                ("model_diagnostics", diagnostics),
                ("model_coefficients", allcoef),
            ]:
                pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    import runpy

    runpy.run_path(str(Path(__file__).resolve().parent / "figures_courtship.py"))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
