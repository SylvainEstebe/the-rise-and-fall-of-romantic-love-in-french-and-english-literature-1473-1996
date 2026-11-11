"""Final ordinal shape robustness: no spline selection, full sampling, retry and LOO audit.

Tested: bambi 0.17.2, pymc 5.28.5, arviz 0.22.0, formulae 0.6.2,
numpy 2.4.6, pandas 2.3.3. Bambi predict uses response_params (legacy: mean).
Run this file with --help. No import from the exploratory analysis is required.

LOO: use exact az.reloo for manageable flagged sets, with fixed basis/priors and
proper singleton-author integration. Also exclude the UNION of raw k>0.7 rows
from scoring every model as a sensitivity check. That score exclusion itself
is not exact reloo and does not delete records from posterior fitting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/love_spline_mpl")
os.environ.setdefault("PYTENSOR_FLAGS", "compiledir=/tmp/love_spline_pytensor")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis import runtime

runtime.configure()  # Apply the pinned environment’s macOS compiler fix.

# isort: off
# Import order is intentional: initialize Matplotlib before ArviZ's styles.
import matplotlib

matplotlib.use("Agg")
import matplotlib.style.core  # Needed by arviz-plots with some Matplotlib versions.
import matplotlib.pyplot as plt
import arviz as az
import bambi as bmb
import formulae
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import log_expit, logsumexp
import xarray as xr
# isort: on

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results/courtship"
DATASETS = {
    "french_books": (DATA_DIR / "roland_fr_data.csv", 1879, (1500, 1996)),
    "english_books": (DATA_DIR / "novels_uk_data.csv", 1887, (1473, 1949)),
    "french_theatre": (DATA_DIR / "theatre_fr_data.csv", 1060, (1574, 1899)),
    "english_theatre": (DATA_DIR / "theatre_uk_data.csv", 932, (1576, 1800)),
    "french_babel": (DATA_DIR / "babel_fr_data.csv", 1870, (1500, 2000)),
}
FORMULAS = {
    "linear": "score ~ year_c + (1|author)",
    "quadratic": "score ~ year_c + I(year_c**2) + (1|author)",
    **{
        f"spline_{k}": f"score ~ bs(year_c, df={k}, degree=3) + (1|author)"
        for k in range(3, 7)
    },
}


def prepare_data(data):
    """Drop missing outcomes only; reject invalid years, scores or author IDs."""
    raw = pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()
    required = {"year", "score", "author"}
    if not required <= set(raw):
        raise ValueError(f"Missing columns: {required - set(raw)}")
    n_missing = int(raw.score.isna().sum())
    d = raw.loc[raw.score.notna()].copy().reset_index(drop=True)
    for col in ("year", "score"):
        d[col] = pd.to_numeric(d[col], errors="raise")
        if not np.isfinite(d[col]).all():
            raise ValueError(f"Nonfinite {col}")
    if not len(d) or not d.score.isin(range(7)).all():
        raise ValueError("Scores must be integers 0,...,6; no rounding is performed.")
    if not np.equal(d.year, np.floor(d.year)).all() or d.year.nunique() < 7:
        raise ValueError(
            "Need integer publication years and at least seven distinct years."
        )
    if d.author.isna().any() or d.author.astype(str).str.strip().eq("").any():
        raise ValueError(
            "Supply preprocessed, nonmissing author IDs (unique IDs for unknowns)."
        )
    d["author"] = d.author.astype(str)
    # Fail explicitly rather than silently changing the ordinal scale.
    if set(d.score.unique()) != set(range(7)):
        raise ValueError("This analysis expects all seven score levels to be observed.")
    d["score"] = pd.Categorical(d.score.astype(int), categories=range(7), ordered=True)
    mean_year = float(d.year.mean())
    d["year_c"] = (d.year - mean_year) / 100
    return d, mean_year, n_missing


def expected_scores(
    model, idata, years, mean_year, chunk_size=64, *, prediction_covariates=None
):
    """Array [chain*draw, year]; author's latent deviation is fixed at zero.

    Bambi's predict calls evaluate_new_data on the *training* formulae design.
    Small batches avoid constructing a large [chain, draw, year, category] array.
    This does not integrate over the distribution of author random effects.
    """
    kind_default = inspect.signature(model.predict).parameters["kind"].default
    kind = "mean" if kind_default == "mean" else "response_params"
    # No likelihood or observed data in the temporary prediction object. Keep the
    # original idata (and its training likelihood) untouched for LOO and storage.
    posterior = idata.posterior.drop_vars(
        ["p", "score_p", "1|author", "1|author_sigma", "1|author_offset"],
        errors="ignore",
    )
    parts = []
    for start in range(0, len(years), chunk_size):
        ys = np.asarray(years[start : start + chunk_size])
        new = pd.DataFrame({"year_c": (ys - mean_year) / 100})
        # Scalar covariates are held fixed across all prediction years.
        for name, value in (prediction_covariates or {}).items():
            if name == "year_c" or not np.isscalar(value):
                raise ValueError(
                    "Prediction covariates must be scalar and cannot override year_c"
                )
            new[name] = value
        predicted = az.InferenceData(posterior=posterior.copy(deep=True))
        model.predict(
            predicted, data=new, kind=kind, include_group_specific=False, inplace=True
        )
        names = [n for n in ("p", "score_p") if n in predicted.posterior]
        if len(names) != 1:
            raise RuntimeError(
                f"Cannot identify category probabilities: {list(predicted.posterior)}"
            )
        p = predicted.posterior[names[0]]
        other = [dim for dim in p.dims if dim not in ("chain", "draw")]
        category = [dim for dim in other if "dim" in dim and p.sizes[dim] == 7]
        if len(category) != 1:
            raise RuntimeError(f"Unexpected ordinal probability dimensions: {p.dims}")
        category = category[0]
        observation = [dim for dim in other if dim != category]
        if len(observation) != 1:
            raise RuntimeError(f"Unexpected observation dimensions: {p.dims}")
        labels = np.asarray(p.coords[category], dtype=float)
        np.testing.assert_array_equal(labels, np.arange(7))
        np.testing.assert_allclose(p.sum(category), 1.0, atol=1e-8)
        if not np.isfinite(p).all() or float(p.min()) < -1e-10:
            raise RuntimeError("Invalid category probabilities")
        e = (p * p.coords[category].astype(float)).sum(category)
        parts.append(
            e.stack(sample=("chain", "draw")).transpose("sample", observation[0]).values
        )
    return np.concatenate(parts, axis=1)


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(median=np.nan, lower_95=np.nan, upper_95=np.nan)
    low, med, high = np.quantile(values, [0.025, 0.5, 0.975])
    return dict(median=float(med), lower_95=float(low), upper_95=float(high))


def summarize_shape(expected, years):
    """Draw-wise contrasts, global maximum, and first subsequent threshold crossing."""
    years = np.asarray(years)
    contrasts = []
    for requested_start, requested_end in [(1500, 1650), (1650, 1800), (1800, 1950)]:
        start, end = np.clip([requested_start, requested_end], years[0], years[-1])
        valid = end > start
        values = (
            (
                expected[:, np.searchsorted(years, end)]
                - expected[:, np.searchsorted(years, start)]
            )
            if valid
            else np.array([])
        )
        contrasts.append(
            {
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_start": int(start),
                "actual_end": int(end),
                "endpoints_adjusted": bool(
                    start != requested_start or end != requested_end
                ),
                "status": "estimable"
                if valid
                else "not estimable: no covered interval",
                **interval(values),
            }
        )
    peak_index = expected.argmax(axis=1)  # Earliest year in an exact tie.
    peak = expected[np.arange(len(expected)), peak_index]
    peak_year = years[peak_index]
    maxima = {
        **interval(peak_year),
        "proportion_at_first_year": float((peak_index == 0).mean()),
        "proportion_at_last_year": float((peak_index == len(years) - 1).mean()),
        "tie_rule": "earliest grid year at the global maximum",
    }
    per_draw = pd.DataFrame({"peak_year": peak_year, "peak_expected_score": peak})
    declines = []
    for delta in (0.25, 0.5):
        crossed = (np.arange(len(years))[None, :] > peak_index[:, None]) & (
            expected <= peak[:, None] - delta
        )
        occurs = crossed.any(axis=1)
        onset = np.where(occurs, years[crossed.argmax(axis=1)], np.nan)
        per_draw[f"decline_{delta:g}_year"] = onset
        declines.append(
            {
                "drop_points": delta,
                **interval(onset),
                "proportion_never_within_range": float((~occurs).mean()),
                "n_crossing_draws": int(occurs.sum()),
                "n_draws": len(occurs),
                "interval_condition": "conditional on crossing within observed range",
            }
        )
    return (
        pd.DataFrame(contrasts),
        pd.DataFrame([maxima]),
        pd.DataFrame(declines),
        per_draw,
    )


def diagnostics(idata):
    names = [n for n in idata.posterior if n not in ("p", "score_p")]
    table = az.summary(idata, var_names=names, kind="diagnostics", round_to="none")
    row = {
        "divergences": int(idata.sample_stats.diverging.sum()),
        "max_rhat": float(table.r_hat.max()),
        "min_ess_bulk": float(table.ess_bulk.min()),
        "min_ess_tail": float(table.ess_tail.min()),
        "nonfinite_diagnostic_cells": int((~np.isfinite(table)).sum().sum()),
    }
    row["fails_requested_checks"] = bool(
        row["divergences"]
        or row["max_rhat"] > 1.01
        or row["min_ess_bulk"] < 400
        or row["nonfinite_diagnostic_cells"]
    )
    return row, table


def fit_checked(
    model,
    out,
    name,
    *,
    draws,
    tune,
    chains,
    cores,
    seed,
    target_accept,
    save_posterior=True,
):
    """One initial attempt, and one full refit at 4,000 tune/.95 if required."""
    path = out / f"{name}.nc"
    record_path = out / f"{name}_attempts.json"
    attempts = json.loads(record_path.read_text()) if record_path.exists() else []
    if path.exists():
        idata = az.from_netcdf(path)
        idata.load()
        idata.close()
        return idata, attempts
    # A failed initial attempt's diagnostics suffice to resume directly at retry.
    start_attempt = 1 if attempts and attempts[0]["fails_requested_checks"] else 0
    if start_attempt == 0:
        attempts = []
    for attempt in range(start_attempt, 2):
        actual_tune = tune if attempt == 0 else max(4000, tune)
        actual_accept = target_accept if attempt == 0 else max(0.95, target_accept)
        started = time.monotonic()
        print(
            f"FIT {name} attempt={attempt + 1} tune={actual_tune} "
            f"draws={draws} chains={chains} target_accept={actual_accept}",
            flush=True,
        )
        kwargs = dict(
            draws=draws,
            tune=actual_tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            target_accept=actual_accept,
            idata_kwargs={"log_likelihood": True},
            progressbar=False,
        )
        include = (
            "include_response_params"
            if "include_response_params" in inspect.signature(model.fit).parameters
            else "include_mean"
        )
        kwargs[include] = False
        idata = model.fit(**kwargs)
        diag, parameters = diagnostics(idata)
        entry = {
            "attempt": attempt + 1,
            "tune": actual_tune,
            "target_accept": actual_accept,
            "draws_per_chain": draws,
            "seconds": time.monotonic() - started,
            **diag,
        }
        attempts = [a for a in attempts if a["attempt"] < attempt + 1] + [entry]
        record_path.write_text(json.dumps(attempts, indent=2))
        parameters.to_csv(out / f"{name}_attempt_{attempt + 1}_diagnostics.csv")
        if diag["fails_requested_checks"] and attempt == 0:
            print(f"RETRY {name}: {diag}", flush=True)
            del idata
            gc.collect()
            continue
        parameters.to_csv(out / f"{name}_parameter_diagnostics.csv")
        if save_posterior:
            temporary = path.with_suffix(".partial.nc")
            idata.to_netcdf(temporary)
            temporary.replace(path)
        return idata, attempts
    raise RuntimeError("Unexpected sampling control flow")


def fixed_basis_refit_model(data, model_name, excluded):
    """Keep full-data centering, knot positions, boundaries and all prior scales."""
    base = bmb.Model(FORMULAS[model_name], data, family="cumulative", link="logit")
    train = data.drop(index=excluded).copy()
    formula, namespace = FORMULAS[model_name], {}
    if model_name.startswith("spline"):
        k = int(model_name.split("_")[-1])
        namespace = {
            "fixed_knots": np.percentile(
                data.year_c, 100 * np.linspace(0, 1, k - 1)[1:-1]
            ),
            "lower": float(data.year_c.min()),
            "upper": float(data.year_c.max()),
        }
        formula = (
            f"score ~ bs(year_c, df={k}, degree=3, knots=fixed_knots, "
            "lower_bound=lower, upper_bound=upper) + (1|author)"
        )
    model = bmb.Model(
        formula,
        train,
        family="cumulative",
        link="logit",
        auto_scale=False,
        extra_namespace=namespace,
    )
    priors = {
        new.name: copy.deepcopy(old.prior)
        for old, new in zip(
            base.components["p"].common_terms.values(),
            model.components["p"].common_terms.values(),
        )
    }
    priors["1|author"] = copy.deepcopy(
        base.components["p"].group_specific_terms["1|author"].prior
    )
    priors["threshold"] = copy.deepcopy(base.components["threshold"].prior)
    model.set_priors(priors)
    mask = np.arange(len(data)) != excluded
    np.testing.assert_allclose(
        base.components["p"].design.common.design_matrix[mask],
        model.components["p"].design.common.design_matrix,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(model.response_component.term.levels, dtype=int), np.arange(7)
    )
    return model, train


def heldout_log_probability(model, idata, heldout, *, unseen_author=False):
    """Conditional held-out log p; integrate a singleton author's normal prior.

    Sampling a random *existing* author's posterior would be incorrect for an
    author with no remaining observations. Gauss-Hermite integration instead
    uses N(0, sigma_author) for each refit posterior draw, with a convergence check.
    """
    post = idata.posterior
    count = post.sizes["chain"] * post.sizes["draw"]
    design = (
        model.components["p"].design.common.evaluate_new_data(heldout).design_matrix[0]
    )
    eta, offset = np.zeros(count), 0
    for name in model.components["p"].common_terms:
        beta = (
            post[name]
            .stack(sample=("chain", "draw"))
            .transpose("sample", ...)
            .values.reshape(count, -1)
        )
        eta += beta @ design[offset : offset + beta.shape[1]]
        offset += beta.shape[1]
    assert offset == len(design)
    threshold = (
        post.threshold.stack(sample=("chain", "draw")).transpose("sample", ...).values
    )
    score = int(heldout.score.iloc[0])

    def logp(location):
        t = threshold if location.ndim == 1 else threshold[:, :, None]
        if score == 0:
            return log_expit(t[:, 0] - location)
        if score == 6:
            return log_expit(location - t[:, -1])
        lower, upper = t[:, score - 1], t[:, score]
        return (
            log_expit(upper - location)
            + log_expit(location - lower)
            + np.log(-np.expm1(lower - upper))
        )

    quadrature_error = 0.0
    if unseen_author:
        sigma = post["1|author_sigma"].stack(sample=("chain", "draw")).values
        previous = None
        for nodes_count in (64, 128, 256):
            nodes, weights = np.polynomial.hermite.hermgauss(nodes_count)
            values = logsumexp(
                logp(eta[:, None] + np.sqrt(2) * sigma[:, None] * nodes)
                + np.log(weights)[None, :]
                - 0.5 * np.log(np.pi),
                axis=1,
            )
            score_mean = float(logsumexp(values) - np.log(count))
            if previous is not None:
                quadrature_error = abs(score_mean - previous)
                if quadrature_error < 1e-5:
                    break
            previous = score_mean
        else:
            raise RuntimeError("Singleton-author quadrature did not converge")
    else:
        group = post["1|author"]
        (dim,) = [dim for dim in group.dims if dim not in ("chain", "draw")]
        deviation = (
            group.sel({dim: str(heldout.author.iloc[0])})
            .stack(sample=("chain", "draw"))
            .values
        )
        values = logp(eta + deviation)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite held-out log probability")
    return values.reshape(post.sizes["chain"], post.sizes["draw"]), quadrature_error


class OrdinalReLOOWrapper(az.SamplingWrapper):
    def __init__(self, data, model_name, out, settings):
        super().__init__(model=None)
        self.data, self.model_name, self.out, self.settings = (
            data,
            model_name,
            out,
            settings,
        )

    def check_implemented_methods(self, methods):
        # ArviZ 0.22's base checker actually invokes sample(1). Avoid launching
        # an unintended refit just to test that these overrides exist.
        implemented = {
            "sel_observations",
            "sample",
            "get_inference_data",
            "log_likelihood__i",
        }
        return [method for method in methods if method not in implemented]

    def sel_observations(self, idx):
        row = int(np.asarray(idx).item())
        return row, row

    def sample(self, row):
        out = self.out / "reloo" / self.model_name / f"row_{row}"
        out.mkdir(parents=True, exist_ok=True)
        cache = out / "heldout_log_probability.npz"
        if cache.exists():
            with np.load(cache) as saved:
                values = saved["log_probability"]
        else:
            model, train = fixed_basis_refit_model(self.data, self.model_name, row)
            (out / "model.txt").write_text(str(model))
            idata, attempts = fit_checked(
                model, out, "refit", save_posterior=False, **self.settings
            )
            if attempts[-1]["fails_requested_checks"]:
                raise RuntimeError(f"Exact LOO refit failed diagnostics: {out}")
            heldout = self.data.iloc[[row]].copy()
            unseen = not train.author.eq(heldout.author.iloc[0]).any()
            values, error = heldout_log_probability(
                model, idata, heldout, unseen_author=unseen
            )
            info = {
                "row": row,
                "unseen_author": bool(unseen),
                "quadrature_log_elpd_error": error,
                "loo_i": float(logsumexp(values) - np.log(values.size)),
                "model": self.model_name,
                **attempts[-1],
            }
            (out / "result.json").write_text(json.dumps(info, indent=2))
            np.savez_compressed(cache, log_probability=values)
            del idata, model
            gc.collect()
        return az.InferenceData(
            posterior=xr.Dataset({"heldout_logp": (("chain", "draw"), values)})
        )

    def get_inference_data(self, fitted_model):
        return fitted_model

    def log_likelihood__i(self, excluded_obs, idata__i):
        return idata__i.posterior.heldout_logp


def exact_reloo_dataset(out, cores=2, maximum_refits=12):
    """Repair the available flagged models; safe to run again as fits finish."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    data, _, _ = prepare_data(out / "analysis_data.csv")
    paths = [
        out / f"{name}_loo_pointwise.csv"
        for name in FORMULAS
        if (out / f"{name}_loo_pointwise.csv").exists()
    ]
    flagged = sum(int((pd.read_csv(p).pareto_k > 0.7).sum()) for p in paths)
    if flagged > maximum_refits:
        print(
            f"LOO sensitivity fallback: {flagged} exact refits exceed limit {maximum_refits}",
            flush=True,
        )
        return
    settings = dict(
        draws=manifest["draws"],
        tune=manifest["tune"],
        chains=manifest["chains"],
        cores=cores,
        seed=manifest["seed"],
        target_accept=manifest["initial_target_accept"],
    )
    for name in FORMULAS:
        source = out / f"{name}_loo_pointwise.csv"
        if not source.exists():
            continue
        points = pd.read_csv(source)
        if not (points.pareto_k > 0.7).any():
            continue
        print(
            f"EXACT RELOO {out.name} {name}: {(points.pareto_k > 0.7).sum()} observations",
            flush=True,
        )
        original = az.from_netcdf(out / f"{name}.nc")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo = az.loo(original, pointwise=True, scale="log", var_name="score")
        original.close()
        del original
        np.testing.assert_allclose(loo.loo_i, points.loo_i, atol=1e-8)
        corrected = az.reloo(
            OrdinalReLOOWrapper(data, name, out, settings),
            loo_orig=loo,
            k_thresh=0.7,
            verbose=False,
        )
        points["psis_loo_i"] = points.loo_i
        points["loo_i"] = np.asarray(corrected.loo_i)
        points["refitted_loo"] = points.pareto_k > 0.7
        points.to_csv(out / f"{name}_reloo_pointwise.csv", index=False)


def compare_pointwise(points, mask=None):
    """ELPD and paired difference SE on identical evaluation rows for all models."""
    n = len(next(iter(points.values())))
    keep = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if keep.shape != (n,) or keep.sum() < 2:
        raise ValueError("Comparison requires at least two common evaluation rows")
    for p in points.values():
        np.testing.assert_array_equal(p.row, np.arange(n))
    scores = {name: p.loc[keep, "loo_i"].to_numpy() for name, p in points.items()}
    best = max(scores, key=lambda name: scores[name].sum())
    rows = []
    for name, score in scores.items():
        diff = scores[best] - score
        gap = float(diff.sum())
        dse = float(np.sqrt(len(diff) * np.var(diff, ddof=0)))
        rows.append(
            {
                "model": name,
                "reference_best": best,
                "n_scored": int(keep.sum()),
                "n_excluded": int((~keep).sum()),
                "elpd_loo": float(score.sum()),
                "elpd_se": float(np.sqrt(len(score) * np.var(score, ddof=0))),
                "delta_elpd_loss_vs_best": gap,
                "delta_elpd_se": dse,
                "equivalent_2se": bool(name == best or abs(gap) < 2 * dse),
                "p_loo": float((points[name].loc[keep, "lppd_i"] - score).sum())
                if "lppd_i" in points[name]
                else np.nan,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("elpd_loo", ascending=False)
        .reset_index(drop=True)
    )


def comparison_tables(source, out):
    raw = {name: pd.read_csv(source / f"{name}_loo_pointwise.csv") for name in FORMULAS}
    compare_pointwise(raw).to_csv(out / "loo_psis_uncorrected.csv", index=False)
    points = {}
    for name, p in raw.items():
        repaired = source / f"{name}_reloo_pointwise.csv"
        if repaired.exists():
            q = pd.read_csv(repaired)
            np.testing.assert_array_equal(q.row, p.row)
            np.testing.assert_allclose(q.psis_loo_i, p.loo_i, atol=1e-8)
            points[name] = q
        else:
            points[name] = p
    n = len(points["linear"])
    bad = np.zeros(n, dtype=bool)
    for p in points.values():
        if len(p) != n or not np.isfinite(p.loo_i).all():
            raise ValueError("Invalid pointwise LOO results")
        np.testing.assert_array_equal(p.row, points["linear"].row)
        for col in ("id", "year", "author", "score"):
            if col in p and col in points["linear"]:
                np.testing.assert_array_equal(p[col], points["linear"][col])
        bad |= (p.pareto_k.to_numpy() > 0.7) | ~np.isfinite(p.pareto_k)
    full = compare_pointwise(points)
    reduced = compare_pointwise(points, ~bad)
    extra = pd.DataFrame(
        [
            {
                "model": name,
                "n_pareto_k_gt_0_7": int((p.pareto_k > 0.7).sum()),
                "n_exact_loo_refits": int(p.refitted_loo.sum())
                if "refitted_loo" in p
                else 0,
                "max_pareto_k": float(p.pareto_k.max()),
            }
            for name, p in points.items()
        ]
    )
    full = full.merge(extra, on="model", validate="one_to_one")
    full.to_csv(out / "loo_full.csv", index=False)
    reduced.to_csv(out / "loo_without_flagged_rows.csv", index=False)
    flagged = (
        points["linear"]
        .loc[
            bad,
            [
                c
                for c in ("row", "id", "year", "author", "score")
                if c in points["linear"]
            ],
        ]
        .copy()
    )
    for name, p in points.items():
        flagged[f"pareto_k_{name}"] = p.loc[bad, "pareto_k"].to_numpy()
    flagged.to_csv(out / "loo_flagged_rows_union.csv", index=False)
    return full, reduced


def long_shape_table(expected, years, model_name):
    contrasts, peak, onset, draws = summarize_shape(expected, years)
    contrasts["quantity"] = [
        f"contrast_{a}_{b}"
        for a, b in zip(contrasts.requested_start, contrasts.requested_end)
    ]
    peak["quantity"] = "global_maximum_year"
    onset["quantity"] = [f"decline_{v:g}_year" for v in onset.drop_points]
    result = pd.concat([contrasts, peak, onset], ignore_index=True)
    result.insert(0, "model", model_name)
    result["spline_df"] = (
        int(model_name.split("_")[-1]) if model_name.startswith("spline") else np.nan
    )
    result["interior_knots"] = result.spline_df - 3
    return result, draws


def plot_overlay(data, curves, out, title):
    observed = data.assign(
        score=pd.to_numeric(data.score), bin=20 * np.floor(data.year / 20)
    )
    bins = observed.groupby("bin").agg(
        year=("year", "mean"),
        n=("score", "size"),
        mean=("score", "mean"),
        sd=("score", "std"),
    )
    bins = bins.loc[bins.n >= 10].copy()
    bins["se"] = bins.sd / np.sqrt(bins.n)
    bins.to_csv(out / "twenty_year_means.csv")
    plot_low = min(0.0, float((bins["mean"] - 1.96 * bins.se).min())) - 0.1
    plot_high = max(6.0, float((bins["mean"] + 1.96 * bins.se).max())) + 0.1
    colors = {
        "spline_3": "#444444",
        "spline_4": "#228833",
        "spline_5": "#cc6677",
        "spline_6": "#aa44aa",
        "quadratic": "#111111",
    }
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for name in ["spline_3", "spline_4", "spline_5", "spline_6", "quadratic"]:
        q = curves[curves.model.eq(name)]
        if name != "quadratic":
            ax.fill_between(
                q.year, q.lower_95, q.upper_95, color=colors[name], alpha=0.075
            )
        label = (
            "Quadratic"
            if name == "quadratic"
            else (
                "df=3 (global cubic)" if name == "spline_3" else f"Spline df={name[-1]}"
            )
        )
        ax.plot(
            q.year,
            q["mean"],
            color=colors[name],
            lw=1.9,
            ls="--" if name == "quadratic" else "-",
            label=label,
        )
    ax.errorbar(
        bins.year,
        bins["mean"],
        yerr=1.96 * bins.se,
        fmt="o",
        ms=4,
        color="#2277bb",
        capsize=2,
        label="20-year means ± 1.96 SE (n ≥ 10)",
    )
    ax.set(
        title=title,
        xlabel="Publication year",
        ylabel="Expected courtship score (0–6)",
        xlim=(data.year.min(), data.year.max()),
        ylim=(plot_low, plot_high),
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"all_shapes.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.15
        )
    plt.close(fig)


def summarize_dataset(source, out, name):
    """Recompute all quantities from every draw; also works on the saved pilot."""
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    full, reduced = comparison_tables(source, out)
    tables, curves = [], []
    for model_name in list(FORMULAS)[1:]:
        with np.load(source / f"{model_name}_expected.npz") as saved:
            expected, years = saved["expected"], saved["years"]
            table, draws = long_shape_table(expected, years, model_name)
            tables.append(table)
            draws.to_csv(out / f"{model_name}_shape_draws.csv", index=False)
            low, high = np.quantile(expected, [0.025, 0.975], axis=0)
            curves.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "year": years,
                        "mean": expected.mean(axis=0),
                        "lower_95": low,
                        "upper_95": high,
                    }
                )
            )
    long = pd.concat(tables, ignore_index=True)
    cols = [
        "model",
        "reference_best",
        "delta_elpd_loss_vs_best",
        "delta_elpd_se",
        "equivalent_2se",
        "elpd_loo",
        "elpd_se",
        "n_scored",
        "n_excluded",
    ]
    long = long.merge(full, on="model", validate="many_to_one")
    long = long.merge(
        reduced[cols].rename(
            columns={c: f"without_flagged_{c}" for c in cols if c != "model"}
        ),
        on="model",
        validate="many_to_one",
    )
    long.insert(0, "dataset", name)
    long.to_csv(out / "all_model_quantities.csv", index=False)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "all_curves.csv", index=False)
    plot_overlay(
        pd.read_csv(source / "analysis_data.csv"),
        curves,
        out,
        name.replace("_", " ").title(),
    )
    if source != out:
        for filename in ("manifest.json", "diagnostics.csv"):
            (out / filename).write_bytes((source / filename).read_bytes())
    return long, full, reduced


def run_dataset(
    data,
    name,
    output_dir,
    *,
    draws=4000,
    tune=2000,
    chains=4,
    cores=2,
    seed=0,
    target_accept=0.8,
    expected_n=None,
    expected_range=None,
):
    d, mean_year, n_missing = prepare_data(data)
    years = np.arange(int(d.year.min()), int(d.year.max()) + 1)
    if expected_n is not None and len(d) != expected_n:
        raise ValueError(f"Unexpected sample size for {name}")
    if expected_range is not None and tuple(years[[0, -1]]) != tuple(expected_range):
        raise ValueError(f"Unexpected year range for {name}")
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_sha256": hashlib.sha256(
            d[["score", "year", "author"]].to_csv(index=False).encode()
        ).hexdigest(),
        "n": len(d),
        "n_authors": d.author.nunique(),
        "mean_year": mean_year,
        "range": years[[0, -1]].tolist(),
        "dropped_missing_scores": n_missing,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "initial_target_accept": target_accept,
        "retry_tune": max(4000, tune),
        "retry_target_accept": max(0.95, target_accept),
        "formulas": FORMULAS,
        "family": "cumulative",
        "link": "logit",
        "priors": "Bambi defaults",
        "bambi": bmb.__version__,
        "pymc": pm.__version__,
        "arviz": az.__version__,
        "formulae": formulae.__version__,
        "loo_strategy": "common_flagged_union_exclusion",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Settings/data changed for {out}; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    d.to_csv(out / "analysis_data.csv", index=False)
    loos, diagnostic_rows = {}, []
    for name_model, formula in FORMULAS.items():
        print(f"{name}: {name_model}", flush=True)
        model = bmb.Model(formula, d, family="cumulative", link="logit")
        (out / f"{name_model}_model.txt").write_text(str(model))
        idata, attempts = fit_checked(
            model,
            out,
            name_model,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            seed=seed,
            target_accept=target_accept,
        )
        diag, _ = diagnostics(idata)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loo = az.loo(idata, pointwise=True, scale="log", var_name="score")
        loos[name_model] = loo
        k = np.asarray(loo.pareto_k).ravel()
        row = {
            "dataset": name,
            "model": name_model,
            **diag,
            "n_pareto_k_gt_0_7": int((k > 0.7).sum()),
            "max_pareto_k": float(k.max()),
            "refitted": len(attempts) > 1,
            "final_tune": attempts[-1]["tune"],
            "final_target_accept": attempts[-1]["target_accept"],
            "draws_per_chain": draws,
            "chains": chains,
            "sampling_seconds": sum(a["seconds"] for a in attempts),
        }
        diagnostic_rows.append(row)
        pd.DataFrame(diagnostic_rows).to_csv(out / "diagnostics.csv", index=False)
        ll = idata.log_likelihood["score"]
        (obs_dim,) = [dim for dim in ll.dims if dim not in ("chain", "draw")]
        lppd = []
        for first in range(0, len(d), 128):
            block = ll.isel({obs_dim: slice(first, first + 128)}).transpose(
                "chain", "draw", obs_dim
            )
            values = np.asarray(block).reshape(-1, block.sizes[obs_dim])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite retained log-likelihood")
            lppd.extend(logsumexp(values, axis=0) - np.log(len(values)))
        pointwise = d[[c for c in ("id", "year", "author", "score") if c in d]].copy()
        pointwise.insert(0, "row", np.arange(len(d)))
        pointwise["loo_i"] = np.asarray(loo.loo_i).ravel()
        pointwise["pareto_k"] = k
        pointwise["lppd_i"] = lppd
        pointwise.to_csv(out / f"{name_model}_loo_pointwise.csv", index=False)
        (out / f"{name_model}_loo_warnings.json").write_text(
            json.dumps([str(w.message) for w in caught], indent=2)
        )
        if name_model != "linear":
            expected = expected_scores(model, idata, years, mean_year)
            ix = np.array([0, len(years) // 2, len(years) - 1])
            check = expected_scores(model, idata, years[ix], mean_year, chunk_size=1)
            np.testing.assert_allclose(check, expected[:, ix], atol=1e-10, rtol=1e-10)
            np.savez_compressed(
                out / f"{name_model}_expected.npz", expected=expected, years=years
            )
            del expected
        print(f"DONE {name_model}: {row}", flush=True)
        del ll, block, values, idata, model
        gc.collect()
    az_comparison = az.compare(loos, ic="loo", scale="log")
    az_comparison.to_csv(out / "arviz_compare_full.csv")
    summarize_dataset(out, out, name)
    full_raw = pd.read_csv(out / "loo_psis_uncorrected.csv")
    np.testing.assert_allclose(
        full_raw.set_index("model").loc[az_comparison.index, "elpd_loo"],
        az_comparison.elpd_loo,
    )
    np.testing.assert_allclose(
        full_raw.set_index("model").loc[az_comparison.index, "delta_elpd_se"],
        az_comparison.dse,
        atol=1e-7,
    )
    print(f"CORPUS COMPLETE {name}", flush=True)
    return name


def md_table(frame):
    def fmt(v, col):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            precision = 4 if "rhat" in col or "proportion" in col else 2
            return f"{v:.{precision}f}"
        return str(v).replace("|", "/")

    rows = [list(frame.columns), ["---"] * len(frame.columns)]
    rows += [
        [fmt(v, col) for v, col in zip(row, frame.columns)]
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def plain_summary(long, full, reduced):
    full_names = set(full.loc[full.equivalent_2se, "model"])
    reduced_names = set(reduced.loc[reduced.equivalent_2se, "model"])
    pieces = [
        f"The models within two paired SEs of the highest ELPD are "
        f"{', '.join(sorted(full_names))} using all rows, and "
        f"{', '.join(sorted(reduced_names))} after excluding the "
        f"{int(reduced.n_excluded.iloc[0])} commonly flagged rows from scoring."
    ]
    previous_flexible = None
    for label, names in [
        ("full comparison", full_names),
        ("common-row sensitivity comparison", reduced_names),
    ]:
        if label.startswith("common") and names == full_names:
            pieces.append(
                "The equivalent-model set is unchanged by the common-row sensitivity check."
            )
            continue
        shape_names = names.intersection(set(FORMULAS) - {"linear"})
        contrast = long[
            long.model.isin(shape_names) & long.quantity.eq("contrast_1650_1800")
        ]
        includes = contrast.loc[
            (contrast.lower_95 <= 0) & (contrast.upper_95 >= 0), "model"
        ].tolist()
        pieces.append(
            f"In the {label}, the 1650→1800 contrast includes zero for "
            f"{', '.join(includes) if includes else 'none of these curved models'}."
        )
        flexible = shape_names.intersection({"spline_4", "spline_5", "spline_6"})
        if not flexible:
            pieces.append("No df=4–6 spline meets this comparison rule.")
            continue
        if flexible == previous_flexible:
            pieces.append(
                "The equivalent flexible splines, and thus their onset-date ranges, are unchanged."
            )
            continue
        previous_flexible = flexible
        peaks = long[
            long.model.isin(flexible) & long.quantity.eq("global_maximum_year")
        ]
        if peaks.proportion_at_first_year.max() > 0.1:
            pieces.append(
                f"For these flexible models, {100 * peaks.proportion_at_first_year.min():.1f}%–"
                f"{100 * peaks.proportion_at_first_year.max():.1f}% of draws peak at the first "
                "covered year; very early global decline dates can therefore reflect "
                "boundary maxima rather than a decline after the proposed plateau."
            )
        for threshold in (0.25, 0.5):
            sub = long[
                long.model.isin(flexible)
                & long.quantity.eq(f"decline_{threshold:g}_year")
            ]
            finite = sub[sub["median"].notna()]
            if finite.empty:
                pieces.append(
                    f"None of the equivalent flexible-spline draws crosses the {threshold:g}-point threshold."
                )
            else:
                spread = finite["median"].max() - finite["median"].min()
                pieces.append(
                    f"Across equivalent df=4–6 splines, the conditional {threshold:g}-point decline "
                    f"median ranges from {finite['median'].min():.0f} to {finite['median'].max():.0f} "
                    f"(a {spread:.0f}-year shift across df); their 95% intervals collectively span "
                    f"{finite.lower_95.min():.0f}–{finite.upper_95.max():.0f}, and non-crossing "
                    f"proportions range from {100 * sub.proportion_never_within_range.min():.1f}% "
                    f"to {100 * sub.proportion_never_within_range.max():.1f}%."
                )
    pieces.append(
        "Including zero supports compatibility with no net change between the endpoints, "
        "not a demonstration that the intervening curve is flat; decline dates condition "
        "on crossing before the corpus ends."
    )
    return " ".join(pieces)


def compact_quantities(long):
    rows = []
    for r in long.itertuples(index=False):
        digits = 2 if r.quantity.startswith("contrast") else 0
        estimate = (
            f"{r.median:.{digits}f} [{r.lower_95:.{digits}f}, {r.upper_95:.{digits}f}]"
            if pd.notna(r.median)
            else "not estimable / no crossings"
        )
        endpoints = (
            f"{r.actual_start:.0f}→{r.actual_end:.0f}"
            if pd.notna(r.actual_start)
            else "—"
        )
        boundary = (
            f"{100 * r.proportion_at_first_year:.2f}% / {100 * r.proportion_at_last_year:.2f}%"
            if pd.notna(r.proportion_at_first_year)
            else "—"
        )
        never = (
            f"{100 * r.proportion_never_within_range:.2f}%"
            if pd.notna(r.proportion_never_within_range)
            else "—"
        )
        rows.append(
            {
                "Model": r.model,
                "Quantity": r.quantity,
                "Actual endpoints": endpoints,
                "Median [95% interval]": estimate,
                "Peak first / last": boundary,
                "Never crosses": never,
                "Full ΔELPD (SE)": f"{r.delta_elpd_loss_vs_best:.2f} ({r.delta_elpd_se:.2f})",
                "Common-row ΔELPD (SE)": f"{r.without_flagged_delta_elpd_loss_vs_best:.2f} "
                f"({r.without_flagged_delta_elpd_se:.2f})",
            }
        )
    return pd.DataFrame(rows)


def build_report(root, dataset_names):
    root = Path(root)
    sections = [
        "# Romantic courtship: shape robustness across all models",
        "No single spline is selected. df=3 is a global cubic with no interior knots; "
        "df=4, 5, 6 have one, two, three interior knots. All intervals are 95% equal-tailed.",
        "ELPD loss is best minus candidate (positive means worse). SE means the paired "
        "SE of that difference. The reference is the highest ELPD among all six models, "
        "including linear. The requested equivalence rule is |ΔELPD| < 2 SE, with the "
        "reference itself included. This is a descriptive rule, not a formal equivalence test.",
        "**LOO handling:** exact reloo was not run. The pilot required 71 additional "
        "model–observation fits, implying hours of additional sampling. Both full "
        "PSIS-LOO and common-row sensitivity results are shown. The latter excludes "
        "from scoring the union of rows with k>0.7 in any model, without refitting "
        "the full-data posterior. This is not exact LOO and does not repair the "
        "unreliable full-data ELPD estimates. Rankings are conditional on this limitation.",
    ]
    diagnostic_tables = []
    all_long = []
    attempts = []
    summaries = []
    flagged_pairs = 0
    flagged_rows = 0
    exact_refits = 0
    for name in dataset_names:
        d = root / name
        if not (d / "all_model_quantities.csv").exists():
            continue
        long = pd.read_csv(d / "all_model_quantities.csv")
        full = pd.read_csv(d / "loo_full.csv")
        reduced = pd.read_csv(d / "loo_without_flagged_rows.csv")
        repaired_count = (
            int(full.n_exact_loo_refits.sum()) if "n_exact_loo_refits" in full else 0
        )
        raw_bad_count = int(full.n_pareto_k_gt_0_7.sum())
        (d / "loo_status.json").write_text(
            json.dumps(
                {
                    "primary_method": "PSIS with exact reloo corrections"
                    if repaired_count
                    else "PSIS",
                    "raw_flagged_model_observation_pairs": raw_bad_count,
                    "exact_corrections": repaired_count,
                    "unresolved_flagged_pairs": raw_bad_count - repaired_count,
                    "common_sensitivity_rows_excluded": int(reduced.n_excluded.iloc[0]),
                    "raw_pareto_values_preserved": True,
                },
                indent=2,
            )
        )
        flagged_pairs += int(full.n_pareto_k_gt_0_7.sum())
        flagged_rows += int(reduced.n_excluded.iloc[0])
        exact_refits += (
            int(full.n_exact_loo_refits.sum()) if "n_exact_loo_refits" in full else 0
        )
        diag = pd.read_csv(d / "diagnostics.csv").assign(dataset=name)
        if "n_exact_loo_refits" in full:
            diag = diag.merge(
                full[["model", "n_exact_loo_refits"]], on="model", validate="one_to_one"
            )
        diagnostic_tables.append(diag)
        all_long.append(long)
        manifest = json.loads((d / "manifest.json").read_text())
        summary = plain_summary(long, full, reduced)
        summaries.append(f"**{name.replace('_', ' ').title()}.** {summary}")
        compact = compact_quantities(long)
        diag_columns = [
            c
            for c in (
                "model",
                "divergences",
                "max_rhat",
                "min_ess_bulk",
                "min_ess_tail",
                "n_pareto_k_gt_0_7",
                "n_exact_loo_refits",
                "refitted",
                "final_tune",
                "final_target_accept",
            )
            if c in diag
        ]
        sections += [
            f"## {name.replace('_', ' ').title()}",
            f"n={manifest['n']}, years {manifest['range'][0]}–{manifest['range'][1]}; "
            f"{manifest['chains']} chains × {manifest['draws']} retained draws. "
            "Retry settings, where used, are recorded in the diagnostics table.",
            summary,
            "### All model quantities",
            md_table(compact),
            "Decline intervals condition on crossing. Blank contrast dates/intervals "
            "with equal adjusted endpoints mean not estimable, not zero change.",
            "### Full LOO (exact corrections where available)",
            md_table(full),
            f"[Uncorrected PSIS-LOO audit table]({name}/loo_psis_uncorrected.csv)",
            "### Same flagged observations excluded from every model's score",
            md_table(reduced),
            "### Diagnostics",
            md_table(diag[diag_columns]),
            f"![All five curves]({name}/all_shapes.png)",
        ]
        for path in d.glob("*_attempts.json"):
            attempts += [
                {
                    "dataset": name,
                    "model": path.name.removesuffix("_attempts.json"),
                    **a,
                }
                for a in json.loads(path.read_text())
            ]
    sections += [
        "## Interpretation and API details",
        "Expected score is Σ k·P(score=k), computed per draw with author deviations fixed "
        "at zero. It is not an average over the random-effect distribution. Predictions "
        "reuse the fitted formulae transform and original corpus centering. Grid years "
        "are limited to observed endpoints. Bands are pointwise posterior uncertainty "
        "in the mean, conditional on each model; bin bars are ordinary record-level SEs. "
        "Exact maximum ties use the earliest year. A global maximum at a boundary, or "
        "no crossing within range, must not be interpreted as a dated interior turning point.",
        "The script uses the installed InferenceData stack (Bambi 0.17.2/PyMC 5.28.5/"
        "ArviZ 0.22.0/formulae 0.6.2) and the current response_params prediction argument. "
        "Sources: [Bambi Model API](https://bambinos.github.io/bambi/api/Model.html), "
        "[ArviZ compare](https://python.arviz.org/en/v0.22.0/api/generated/arviz.compare.html), "
        "[ArviZ reloo](https://python.arviz.org/en/v0.22.0/api/generated/arviz.reloo.html).",
    ]
    if flagged_pairs == 0:
        sections[3] = (
            "**LOO handling:** every fitted model has zero observations with Pareto k>0.7. "
            "No exact leave-one-out refits were necessary. The full PSIS-LOO and "
            "common-row sensitivity comparison tables are identical."
        )
    elif exact_refits == flagged_pairs:
        sections[3] = (
            f"**LOO handling:** {flagged_pairs} model–observation pairs had raw Pareto k>0.7 "
            f"({flagged_rows} distinct observations across corpora). All were corrected "
            "by exact leave-one-out refitting with az.reloo. Primary full comparisons "
            "combine those refits with reliable PSIS contributions. Raw k values remain "
            "in diagnostics for transparency. The sensitivity comparison excludes the "
            "same originally flagged rows from every model's score. Original PSIS-only "
            "comparisons are also saved. Refit knots, boundaries, centering and priors "
            "are held at full-data values. Singleton-author predictions integrate the "
            "normal random-intercept prior using checked Gauss-Hermite quadrature."
        )
    else:
        sections[3] = (
            f"**LOO handling:** {flagged_pairs} model–observation pairs were flagged, "
            f"covering {flagged_rows} distinct rows across corpora. {exact_refits} were "
            f"corrected with exact az.reloo refits; {flagged_pairs - exact_refits} remain "
            "uncorrected under the configured refit limit/LOO strategy. The primary "
            "full comparison therefore remains provisional. The common-row sensitivity "
            "excludes the union of raw k>0.7 rows from scoring every model, with the "
            "full-data posterior unchanged. This sensitivity is not exact LOO. The "
            "uncorrected PSIS-only audit comparison is also saved."
        )
    (root / "REPORT.md").write_text("\n\n".join(sections) + "\n")
    (root / "dataset_summaries.md").write_text("\n\n".join(summaries) + "\n")
    if diagnostic_tables:
        all_diag = pd.concat(diagnostic_tables, ignore_index=True)
        all_diag.to_csv(root / "diagnostics_all_models.csv", index=False)
        short_cols = [
            c
            for c in (
                "dataset",
                "model",
                "divergences",
                "max_rhat",
                "min_ess_bulk",
                "min_ess_tail",
                "n_pareto_k_gt_0_7",
                "n_exact_loo_refits",
                "refitted",
            )
            if c in all_diag
        ]
        all_diag[short_cols].to_csv(root / "diagnostics_short.csv", index=False)
        pd.concat(all_long, ignore_index=True).to_csv(
            root / "all_model_quantities.csv", index=False
        )
    if attempts:
        pd.DataFrame(attempts).to_csv(root / "sampling_attempts.csv", index=False)
    reloo_rows = [
        json.loads(p.read_text()) | {"dataset": p.parents[3].name}
        for p in root.glob("*/reloo/*/row_*/result.json")
    ]
    if reloo_rows:
        pd.DataFrame(reloo_rows).to_csv(root / "exact_loo_refits.csv", index=False)
    return root / "REPORT.md"


def dataset_worker(name, config):
    out = Path(config["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            path, n, limits = DATASETS[name]
            return run_dataset(
                path, name, expected_n=n, expected_range=limits, **config
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/splines/affiliation"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument(
        "--summarize-from",
        type=Path,
        help="Only recompute reports from an existing run, including the pilot",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild Markdown from existing tables",
    )
    parser.add_argument(
        "--reloo-only",
        action="store_true",
        help="Repair currently available flagged LOO rows",
    )
    parser.add_argument(
        "--loo-strategy", choices=["auto", "sensitivity"], default="auto"
    )
    parser.add_argument(
        "--max-reloo-refits",
        type=int,
        default=12,
        help="Use the common-row sensitivity fallback above this count per corpus",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        print(build_report(args.output, args.datasets), flush=True)
        return
    if args.reloo_only:
        for name in args.datasets:
            exact_reloo_dataset(args.output / name, args.cores, args.max_reloo_refits)
        return
    if args.summarize_from:
        for name in args.datasets:
            summarize_dataset(args.summarize_from / name, args.output / name, name)
    else:
        config = dict(
            output_dir=args.output,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            seed=args.seed,
            target_accept=args.target_accept,
        )
        if args.workers == 1:
            for name in args.datasets:
                dataset_worker(name, config)
                print(f"Completed {name}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                order = sorted(
                    args.datasets, key=lambda name: DATASETS[name][1], reverse=True
                )
                futures = {
                    executor.submit(dataset_worker, name, config): name
                    for name in order
                }
                for future in as_completed(futures):
                    print(f"Completed {future.result()}", flush=True)
        if args.loo_strategy == "auto":
            for name in args.datasets:
                exact_reloo_dataset(
                    args.output / name, args.cores, args.max_reloo_refits
                )
                summarize_dataset(args.output / name, args.output / name, name)
    print(build_report(args.output, args.datasets), flush=True)


if __name__ == "__main__":
    main()
