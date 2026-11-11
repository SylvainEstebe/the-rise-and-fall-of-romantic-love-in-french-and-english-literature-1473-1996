#!/usr/bin/env python3
"""Requested ING_38 century contrasts, decline-rate timing, and Babel bootstrap.

Reuses the final posterior draws. Bootstrap exactly replays the original seed-42
random stream so the already reported century-change intervals are reproduced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/splines/contrasts"
BASE = ROOT / "results/splines/affiliation"
DECOMP = ROOT / "results/courtship"
SOURCES = {
    "french_books": BASE / "french_books",
    "french_babel": BASE / "french_babel",
    "english_babel": ROOT / "results/splines/affiliation_babel/english_babel",
}
MODELS = ["quadratic", "spline_3", "spline_4", "spline_5", "spline_6"]
B = 3000


def summarize(values):
    low, med, high = np.nanquantile(values, [0.025, 0.5, 0.975])
    return {"median": float(med), "lower_95": float(low), "upper_95": float(high)}


def posterior_summaries():
    contrasts, slopes = [], []
    for dataset, source in SOURCES.items():
        loo = pd.read_csv(source / "loo_full.csv").set_index("model")
        selected = loo.loc[["spline_4", "spline_5", "spline_6"], "elpd_loo"].idxmax()
        windows = [(1800, 1900), (1900, 1996)]
        if dataset == "french_books":
            windows.insert(0, (1700, 1800))
        for model in MODELS:
            with np.load(source / f"{model}_expected.npz") as saved:
                e, years = saved["expected"], saved["years"]
            assert len(e) == 16000 and np.isfinite(e).all()
            assert np.array_equal(years, np.arange(years[0], years[-1] + 1))
            meta = {
                "dataset": dataset,
                "model": model,
                "displayed": model == selected,
                "genuinely_flexible": model in ["spline_4", "spline_5", "spline_6"],
                **loo.loc[model].to_dict(),
            }
            values = []
            for start, end in windows:
                assert start in years and end in years
                delta = (
                    e[:, np.searchsorted(years, end)]
                    - e[:, np.searchsorted(years, start)]
                )
                values.append(delta)
                contrasts.append(
                    {
                        **meta,
                        "start": start,
                        "end": end,
                        **summarize(delta),
                        "p_negative": float((delta < 0).mean()),
                        "p_abs_change_below_0_25": float((np.abs(delta) < 0.25).mean()),
                    }
                )
            if dataset != "french_books":
                continue
            # Primary: most negative one-year change over the full observed range.
            # Ten-year average slopes are a descriptive sensitivity to grid width.
            draw_table = pd.DataFrame()
            for width in (1, 10):
                rate = (e[:, width:] - e[:, :-width]) / width
                idx = rate.argmin(axis=1)  # Earliest interval in an exact tie.
                start = years[idx]
                mid = start + width / 2
                min_rate = rate[np.arange(len(e)), idx]
                avg_rates = np.column_stack(
                    [
                        delta / (end - start_year)
                        for delta, (start_year, end) in zip(values, windows)
                    ]
                )
                row = {
                    **meta,
                    "interval_width_years": width,
                    "date_definition": "start year t of minimum [E(t+w)-E(t)]/w on entire observed range",
                    **{f"start_year_{k}": v for k, v in summarize(start).items()},
                    **{f"midpoint_year_{k}": v for k, v in summarize(mid).items()},
                    **{
                        f"rate_points_per_century_{k}": v
                        for k, v in summarize(min_rate * 100).items()
                    },
                    "p_steepest_interval_midpoint_in_1800_1899": float(
                        ((mid >= 1800) & (mid < 1900)).mean()
                    ),
                    "p_first_interval": float((idx == 0).mean()),
                    "p_last_interval": float((idx == rate.shape[1] - 1).mean()),
                    "p_all_rates_nonnegative": float((min_rate >= 0).mean()),
                    "p_1800_1900_most_negative_period_average": float(
                        (avg_rates.argmin(axis=1) == 1).mean()
                    ),
                }
                slopes.append(row)
                draw_table[f"min_{width}y_interval_start"] = start
                draw_table[f"min_{width}y_rate_per_century"] = min_rate * 100
            draw_table.to_csv(
                OUT / f"french_books_{model}_decline_rate_draws.csv", index=False
            )
    pd.DataFrame(contrasts).to_csv(OUT / "spline_contrasts.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "steepest_decline.csv", index=False)


def cluster_statistics(d, windows, weights):
    codes, groups = pd.factorize(d.author, sort=True)
    a = np.zeros((len(groups), len(windows) * 4))
    for j, (lo, hi) in enumerate(windows):
        hit = d.year.between(lo, hi).values
        positive = (d.score.values > 0) & hit
        for k, v in enumerate(
            [hit, positive, d.score.values * positive, d.score.values * hit]
        ):
            a[:, j * 4 + k] = np.bincount(codes, weights=v, minlength=len(groups))
    v = (weights @ a).reshape(B, len(windows), 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.stack(
            [v[:, :, 1] / v[:, :, 0], v[:, :, 2] / v[:, :, 1], v[:, :, 3] / v[:, :, 0]],
            axis=-1,
        )
    assert np.isfinite(stats).all()
    np.testing.assert_allclose(
        stats[:, :, 0] * stats[:, :, 1], stats[:, :, 2], atol=1e-12
    )
    return stats


def bootstrap_babel():
    rng = np.random.default_rng(42)
    levels, changes = [], []
    old = pd.read_csv(DECOMP / "observed_contrasts.csv")
    order = [
        "roland_fr",
        "novels_uk",
        "theatre_fr",
        "theatre_uk",
        "babel_fr",
        "babel_uk",
    ]
    for name in order:
        d = pd.read_csv(DECOMP / f"{name}_data.csv")
        variants = [("all", d)]
        if name == "roland_fr":
            if "n_chapters" not in d:
                lengths = pd.read_csv(ROOT / "results/length/linked_data.csv")
                d = d.merge(
                    lengths[["id", "n_chapters"]], on="id", validate="one_to_one"
                )
                variants = [("all", d)]
            variants.append(("length_12_32", d[d.n_chapters.between(12, 32)]))
        for variant, z in variants:
            g = z.author.nunique()
            for threshold in (1, 3):
                # Original analysis draws a matrix for 20-year bins, then another
                # for century summaries. Discard unused matrices to replay exactly.
                rng.multinomial(g, np.full(g, 1 / g), size=B)
                weights = rng.multinomial(g, np.full(g, 1 / g), size=B)
                if name not in ["babel_fr", "babel_uk"] or threshold != 1:
                    continue
                windows = [
                    (1700, 1799),
                    (1800, 1899),
                    (1900, min(1996, int(z.year.max()))),
                ]
                draws = cluster_statistics(z, windows, weights)
                points = []
                for j, (lo, hi) in enumerate(windows[:2]):
                    frame = z[z.year.between(lo, hi)]
                    positive = frame[frame.score.gt(0)]
                    point = [
                        len(positive) / len(frame),
                        positive.score.mean(),
                        frame.score.mean(),
                    ]
                    points.append(point)
                    for k, metric in enumerate(
                        ["positive_proportion", "mean_among_positives", "overall_mean"]
                    ):
                        bounds = np.quantile(draws[:, j, k], [0.025, 0.975])
                        levels.append(
                            {
                                "corpus": name,
                                "start": lo,
                                "end": hi,
                                "n": len(frame),
                                "n_positive": len(positive),
                                "n_authors": frame.author.nunique(),
                                "metric": metric,
                                "estimate": point[k],
                                "lower_95": bounds[0],
                                "upper_95": bounds[1],
                            }
                        )
                delta = draws[:, 1, :] - draws[:, 0, :]
                oldrow = old[
                    (old.corpus == name)
                    & (old.variant == "all")
                    & (old.threshold == 1)
                    & (old.early == "1700–1799")
                    & (old.late == "1800–1899")
                ].iloc[0]
                for k, (metric, oldkey) in enumerate(
                    [
                        ("positive_proportion", "p"),
                        ("mean_among_positives", "m"),
                        ("overall_mean", "overall_mean"),
                    ]
                ):
                    bounds = np.quantile(delta[:, k], [0.025, 0.975])
                    estimate = points[1][k] - points[0][k]
                    np.testing.assert_allclose(
                        [estimate, *bounds],
                        [
                            oldrow[f"{oldkey}_change"],
                            oldrow[f"{oldkey}_change_low"],
                            oldrow[f"{oldkey}_change_high"],
                        ],
                        atol=1e-12,
                    )
                    changes.append(
                        {
                            "corpus": name,
                            "early": "1700–1799",
                            "late": "1800–1899",
                            "metric": metric,
                            "estimate": estimate,
                            "lower_95": bounds[0],
                            "upper_95": bounds[1],
                        }
                    )
                # Exact symmetric decomposition on the sampled score scale.
                p0, m0 = points[0][:2]
                p1, m1 = points[1][:2]
                contributions = [
                    (
                        "occurrence_contribution",
                        (p1 - p0) * (m0 + m1) / 2,
                        (draws[:, 1, 0] - draws[:, 0, 0])
                        * (draws[:, 1, 1] + draws[:, 0, 1])
                        / 2,
                    ),
                    (
                        "conditional_importance_contribution",
                        (m1 - m0) * (p0 + p1) / 2,
                        (draws[:, 1, 1] - draws[:, 0, 1])
                        * (draws[:, 1, 0] + draws[:, 0, 0])
                        / 2,
                    ),
                ]
                for metric, estimate, vals in contributions:
                    bounds = np.quantile(vals, [0.025, 0.975])
                    changes.append(
                        {
                            "corpus": name,
                            "early": "1700–1799",
                            "late": "1800–1899",
                            "metric": metric,
                            "estimate": estimate,
                            "lower_95": bounds[0],
                            "upper_95": bounds[1],
                        }
                    )
                np.testing.assert_allclose(
                    contributions[0][2] + contributions[1][2], delta[:, 2], atol=1e-12
                )
                np.savez_compressed(
                    OUT / f"{name}_century_bootstrap_draws.npz",
                    draws=draws,
                    windows=windows,
                )
    pd.DataFrame(levels).to_csv(OUT / "babel_century_levels.csv", index=False)
    pd.DataFrame(changes).to_csv(OUT / "babel_century_changes.csv", index=False)


def write_report():
    def table(frame):
        rows = [list(frame), ["---"] * len(frame.columns)]
        rows.extend(frame.astype(str).values.tolist())
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

    def interval(row, key="median", scale=1, digits=2):
        value = row[key] if isinstance(row, pd.Series) else getattr(row, key)
        if digits == 2 and any(
            0 < abs(v * scale) < 0.005 for v in [row.lower_95, row.upper_95]
        ):
            digits = 3
        return (
            f"{value * scale:.{digits}f} "
            f"[{row.lower_95 * scale:.{digits}f}, {row.upper_95 * scale:.{digits}f}]"
        )

    contrasts = pd.read_csv(OUT / "spline_contrasts.csv")
    slopes = pd.read_csv(OUT / "steepest_decline.csv")
    levels = pd.read_csv(OUT / "babel_century_levels.csv")
    changes = pd.read_csv(OUT / "babel_century_changes.csv")
    report = [
        "# Chiffres complémentaires pour les sections 3.3 et 4.2",
        "",
        "Toutes les estimations portent sur **Romantic affiliation (ING_38)**. "
        "Contrastes = score attendu à la date finale moins celui à la date initiale ; "
        "médiane [intervalle crédible central à 95 %], à partir des 16 000 tirages "
        "finaux, à effet auteur nul. Aucun réajustement. Les modèles ci-dessous "
        "sont tous à moins de deux SE du meilleur ELPD de leur corpus. "
        "La spline df = 3 est un polynôme cubique global ; df = 4–6 sont les splines "
        "flexibles de la comparaison principale. Les intervalles propres à chaque "
        "modèle ne sont pas des intervalles intégrant l’incertitude de sélection.",
        "",
    ]
    for dataset, label in [
        ("french_books", "1. Livres français"),
        ("french_babel", "3a. Babel français"),
        ("english_babel", "3b. Babel anglais"),
    ]:
        rows = []
        for model in MODELS:
            q = contrasts[(contrasts.dataset == dataset) & (contrasts.model == model)]
            row = {"Modèle": model + (" (affiché)" if q.displayed.iloc[0] else "")}
            for r in q.itertuples():
                row[f"{r.start}→{r.end}"] = interval(r)
            rows.append(row)
        report.extend([f"## {label}", "", table(pd.DataFrame(rows)), ""])
    rows = []
    for r in slopes[slopes.interval_width_years.eq(1)].itertuples():
        rows.append(
            {
                "Modèle": r.model,
                "Année t [95 %]": f"{r.start_year_median:.0f} [{r.start_year_lower_95:.0f}, {r.start_year_upper_95:.0f}]",
                "P(baisse maximale au XIXe)": f"{100 * r.p_steepest_interval_midpoint_in_1800_1899:.1f} %",
                "P(premier intervalle)": f"{100 * r.p_first_interval:.1f} %",
                "P(dernier intervalle)": f"{100 * r.p_last_interval:.1f} %",
                "P(taux moyen 1800–1900 le plus négatif)": f"{100 * r.p_1800_1900_most_negative_period_average:.1f} %",
            }
        )
    report.extend(
        [
            "## 2. Date de la baisse la plus forte — livres français",
            "",
            table(pd.DataFrame(rows)),
            "",
            "Définition : minimiser E(t+1) − E(t), séparément dans chaque tirage, "
            "sur 1500–1996. L’année rapportée est le début t de cet intervalle annuel ; "
            "c’est une approximation annuelle de la pente instantanée la plus négative, "
            "et non le maximum ni le franchissement de −0,25 point. La dernière "
            "variation est 1995→1996. Les CSV donnent aussi les milieux des intervalles, "
            "les taux de baisse et une sensibilité sur des différences de dix ans.",
            "",
            "**Interprétation :** « steepest in the nineteenth century » n’est pas "
            "robuste à toutes les spécifications plausibles. La spline df = 6 affichée "
            "place la baisse annuelle maximale au XIXe siècle dans 54,2 % des tirages, "
            "et dans le dernier intervalle dans 32,5 %. La probabilité est 99,2 % "
            "pour df = 4 et 78,5 % pour df = 5. La quadratique, également à moins "
            "de deux SE du meilleur modèle, la place au XXe siècle. Le classement "
            "des taux moyens par période répond à une autre question et tient "
            "compte de la durée de 96 ans pour 1900–1996.",
            "",
            "**Après 1900 :** tous les intervalles des splines incluent zéro pour "
            "1900→1996 dans les trois corpus. Cela ne démontre ni une absence de "
            "baisse ni une stabilisation. Les quadratiques prédisent encore une "
            "baisse dans les trois corpus (pour Babel anglais, l’intervalle "
            "est affiché à trois décimales pour conserver le signe de sa borne supérieure). "
            "Les différences de période entre analyses "
            "quadratiques, splines et modèles à deux composantes doivent rester explicites.",
            "",
        ]
    )
    rows = []
    for corpus in ["babel_fr", "babel_uk"]:
        for year in [1700, 1800]:
            q = levels[(levels.corpus == corpus) & (levels.start == year)].set_index(
                "metric"
            )
            p, m = q.loc["positive_proportion"], q.loc["mean_among_positives"]
            rows.append(
                {
                    "Corpus": corpus,
                    "Période": f"{year}–{year + 99}",
                    "N / N positifs": f"{int(p.n)} / {int(p.n_positive)}",
                    "Proportion positive, % [95 %]": interval(p, "estimate", 100, 1),
                    "Moyenne parmi positifs [95 %]": interval(m, "estimate"),
                }
            )
    report.extend(
        ["## 4. Babel : XVIIIe versus XIXe siècle", "", table(pd.DataFrame(rows)), ""]
    )
    rows = []
    for corpus in ["babel_fr", "babel_uk"]:
        q = changes[changes.corpus == corpus].set_index("metric")
        rows.append(
            {
                "Corpus": corpus,
                "Δ proportion, points de %": interval(
                    q.loc["positive_proportion"], "estimate", 100, 1
                ),
                "Δ moyenne parmi positifs": interval(
                    q.loc["mean_among_positives"], "estimate"
                ),
            }
        )
    report.extend(
        [
            table(pd.DataFrame(rows)),
            "",
            "Estimations observées et intervalles bootstrap percentile à 95 % : "
            "3 000 rééchantillonnages de groupes d’auteurs, seed 42, moyennes "
            "pondérées par le nombre d’observations. Un même tirage sert aux deux "
            "périodes. Le flux aléatoire de l’analyse antérieure est reproduit "
            "exactement, et ses intervalles des différences ont été retrouvés "
            "à 10⁻¹² près. Positif signifie score > 0. Babel anglais ne comprend "
            "que 26 observations au XVIIIe siècle, dont 22 positives.",
            "",
            "La baisse de la moyenne conditionnelle est étayée en français "
            "(−0,59 [−1,10 ; −0,09]), mais pas clairement en anglais. Les changements "
            "de proportion sont incertains dans les deux langues. Les contributions "
            "symétriques de chaque composante sont également fournies dans le CSV ; "
            "elles sont descriptives et non causales.",
            "",
            "## 5. Degrés de liberté affichés dans la Figure 1",
            "",
            table(pd.read_csv(OUT / "figure_1_models.csv")),
            "",
            "Choix par meilleur ELPD parmi df = 4, 5, 6 : livres français **6**, "
            "livres anglais **4**, théâtre français **6**, théâtre anglais **6**.",
            "",
        ]
    )
    text = "\n".join(report)
    start = text.index("## 2. Date")
    end = text.index("## 4. Babel")
    rate_section = text[start:end]
    text = text[:start] + text[end:]
    before = text.index("## 3a.")
    text = text[:before] + rate_section + text[before:]
    (OUT / "REPORT.md").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    posterior_summaries()
    bootstrap_babel()
    figures = pd.read_csv(
        ROOT / "results/splines/figure_data/main/displayed_models.csv"
    )
    figures[["dataset", "displayed_model", "selection_scope"]].to_csv(
        OUT / "figure_1_models.csv", index=False
    )
    sources = [p / f"{m}_expected.npz" for p in SOURCES.values() for m in MODELS]
    sources += [
        DECOMP / f"{n}_data.csv"
        for n in [
            "roland_fr",
            "novels_uk",
            "theatre_fr",
            "theatre_uk",
            "babel_fr",
            "babel_uk",
        ]
    ]
    manifest = {
        "construct": "ING_38 (Romantic affiliation / narrative importance of courtship)",
        "posterior": "saved final fits; four chains × 4000 retained draws; no new fits",
        "prediction": "expected scores; author deviations zero; observed range only",
        "credible_intervals": "equal-tailed 2.5th and 97.5th posterior percentiles",
        "steepest_decline": "minimum annual finite difference, on full 1500–1996 grid; 10-year sensitivity also supplied",
        "bootstrap": {
            "unit": "author groups, resampled over the full corpus; record-weighted means",
            "draws": B,
            "seed": 42,
            "interval": "percentile 2.5–97.5%",
            "validation": "original RNG stream replayed; existing century-change intervals reproduced to 1e-12",
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report()
    print(
        "Completed posterior summaries and exactly reproduced Babel bootstrap change intervals."
    )


if __name__ == "__main__":
    main()
