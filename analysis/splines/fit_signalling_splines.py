#!/usr/bin/env python3
"""Fit df 4–6 cumulative-logit author-intercept splines for TH_03, six corpora.

Uses final paper settings and the same helpers as the ING_38 shape analysis.
Every high Pareto-k observation is repaired with exact leave-one-out refitting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import warnings

import ordinal_shape_robustness as a

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/splines/signalling"
SOURCES = {
    **{
        name: ROOT / "results/signaling/data" / f"{key}_TH_03.csv"
        for name, key in [
            ("french_books", "roland_fr"),
            ("english_books", "novels_uk"),
            ("french_theatre", "theatre_fr"),
            ("english_theatre", "theatre_uk"),
        ]
    },
    **{
        name: ROOT / "results/latent/data" / f"{key}_TH_03.csv"
        for name, key in [("french_babel", "babel_fr"), ("english_babel", "babel_uk")]
    },
}
MODELS = ["spline_4", "spline_5", "spline_6"]


def fit_one(name):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    data, mean, n_missing = a.prepare_data(SOURCES[name])
    years = a.np.arange(data.year.min(), data.year.max() + 1)
    manifest = {
        "construct": "TH_03",
        "source": str(SOURCES[name].relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCES[name].read_bytes()).hexdigest(),
        "n": len(data),
        "n_authors": data.author.nunique(),
        "mean_year": mean,
        "range": years[[0, -1]].tolist(),
        "dropped_missing_scores": n_missing,
        "models": MODELS,
        "draws": 4000,
        "chains": 4,
        "tune": 2000,
        "seed": 0,
        "initial_target_accept": 0.8,
        "retry_tune": 4000,
        "retry_target_accept": 0.95,
        "bambi": a.bmb.__version__,
        "pymc": a.pm.__version__,
        "arviz": a.az.__version__,
        "family": "cumulative",
        "link": "logit",
        "priors": "Bambi defaults",
        "prediction": "author deviation zero; training formulae state reused",
    }
    mp = out / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, "Changed checkpoint provenance"
    mp.write_text(json.dumps(manifest, indent=2))
    data.to_csv(out / "analysis_data.csv", index=False)
    diagnostic_rows, quantities, curves = [], [], []
    for model_name in MODELS:
        print(f"{name}: {model_name}", flush=True)
        model = a.bmb.Model(
            a.FORMULAS[model_name], data, family="cumulative", link="logit"
        )
        (out / f"{model_name}_model.txt").write_text(str(model))
        fit, attempts = a.fit_checked(
            model,
            out,
            model_name,
            draws=4000,
            tune=2000,
            chains=4,
            cores=4,
            seed=0,
            target_accept=0.8,
        )
        diag, _ = a.diagnostics(fit)
        assert not diag["fails_requested_checks"], (
            f"{name}/{model_name} failed diagnostics"
        )
        assert (
            dict(fit.posterior.sizes)["chain"] == 4
            and dict(fit.posterior.sizes)["draw"] == 4000
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loo = a.az.loo(fit, pointwise=True, scale="log", var_name="score")
        (out / f"{model_name}_loo_warnings.json").write_text(
            json.dumps([str(w.message) for w in caught])
        )
        k = a.np.asarray(loo.pareto_k).ravel()
        diagnostic_rows.append(
            {
                "dataset": name,
                "model": model_name,
                **diag,
                "n_pareto_k_gt_0_7": int((k > 0.7).sum()),
                "max_pareto_k": float(k.max()),
                "refitted": len(attempts) > 1,
                "final_tune": attempts[-1]["tune"],
                "final_target_accept": attempts[-1]["target_accept"],
            }
        )
        a.pd.DataFrame(diagnostic_rows).to_csv(out / "diagnostics.csv", index=False)
        ll = fit.log_likelihood["score"]
        (obs,) = [d for d in ll.dims if d not in ("chain", "draw")]
        lppd = []
        for start in range(0, len(data), 128):
            block = ll.isel({obs: slice(start, start + 128)}).transpose(
                "chain", "draw", obs
            )
            values = a.np.asarray(block).reshape(-1, block.sizes[obs])
            assert a.np.isfinite(values).all()
            lppd.extend(a.logsumexp(values, axis=0) - a.np.log(len(values)))
        points = data[["id", "year", "author", "score"]].copy()
        points.insert(0, "row", a.np.arange(len(data)))
        points["loo_i"] = a.np.asarray(loo.loo_i).ravel()
        points["pareto_k"] = k
        points["lppd_i"] = lppd
        points.to_csv(out / f"{model_name}_loo_pointwise.csv", index=False)
        expected = a.expected_scores(model, fit, years, mean)
        ix = a.np.array([0, len(years) // 2, len(years) - 1])
        check = a.expected_scores(model, fit, years[ix], mean, chunk_size=1)
        a.np.testing.assert_allclose(expected[:, ix], check, atol=1e-10, rtol=1e-10)
        a.np.savez_compressed(
            out / f"{model_name}_expected.npz", expected=expected, years=years
        )
        q, draws = a.long_shape_table(expected, years, model_name)
        quantities.append(q)
        draws.to_csv(out / f"{model_name}_shape_draws.csv", index=False)
        curves.append(
            a.pd.DataFrame(
                {
                    "model": model_name,
                    "year": years,
                    "mean": expected.mean(0),
                    "lower_95": a.np.quantile(expected, 0.025, axis=0),
                    "upper_95": a.np.quantile(expected, 0.975, axis=0),
                }
            )
        )
        del ll, block, values, fit, model, expected
        gc.collect()
        print(f"DONE {name}: {model_name}", flush=True)
    a.exact_reloo_dataset(out, cores=4, maximum_refits=100)
    raw = {m: a.pd.read_csv(out / f"{m}_loo_pointwise.csv") for m in MODELS}
    repaired = {
        m: a.pd.read_csv(out / f"{m}_reloo_pointwise.csv")
        if (out / f"{m}_reloo_pointwise.csv").exists()
        else raw[m]
        for m in MODELS
    }
    bad = a.np.logical_or.reduce([p.pareto_k.values > 0.7 for p in raw.values()])
    comparison = a.compare_pointwise(repaired)
    extra = a.pd.DataFrame(
        [
            {
                "model": m,
                "n_pareto_k_gt_0_7": int((p.pareto_k > 0.7).sum()),
                "n_exact_loo_refits": int(p.refitted_loo.sum())
                if "refitted_loo" in p
                else 0,
            }
            for m, p in repaired.items()
        ]
    )
    comparison = comparison.merge(extra, on="model", validate="one_to_one")
    assert (comparison.n_pareto_k_gt_0_7 == comparison.n_exact_loo_refits).all()
    comparison.to_csv(out / "loo_full.csv", index=False)
    a.compare_pointwise(raw).to_csv(out / "loo_psis_uncorrected.csv", index=False)
    a.compare_pointwise(repaired, ~bad).to_csv(
        out / "loo_without_flagged_rows.csv", index=False
    )
    long = a.pd.concat(quantities, ignore_index=True).merge(
        comparison, on="model", validate="many_to_one"
    )
    long.insert(0, "dataset", name)
    long.to_csv(out / "all_model_quantities.csv", index=False)
    a.pd.concat(curves, ignore_index=True).to_csv(out / "all_curves.csv", index=False)
    print(f"CORPUS COMPLETE {name}", flush=True)
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fit_one, SOURCES))
    a.pd.concat([a.pd.read_csv(OUT / n / "diagnostics.csv") for n in SOURCES]).to_csv(
        OUT / "diagnostics_all.csv", index=False
    )
