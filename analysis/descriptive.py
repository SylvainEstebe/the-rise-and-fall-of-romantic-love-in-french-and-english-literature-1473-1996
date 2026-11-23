"""Observed two-part decomposition, with author-cluster bootstrap (seed 42)."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/romantic_length_mpl")
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")

import data_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/courtship"
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = ROOT
META = {
    "roland_fr": "metadata_roland_annotations.csv",
    "novels_uk": "metadata_english_works.csv",
    "theatre_fr": "metadata_theatre_fr.csv",
    "theatre_uk": "metadata_theatre_uk.csv",
}
LABELS = {
    "roland_fr": "French book passages",
    "novels_uk": "English book passages",
    "theatre_fr": "French plays",
    "theatre_uk": "English plays",
    "babel_fr": "French Babel ratings",
    "babel_uk": "English Babel ratings",
}
B = 3000


def load_data():
    data = {}
    sources = []
    for key in META:
        a, m, found = data_source.tables(key, root=PRIMARY)
        sources.extend(found)
        a = a.drop(columns=["date", "author"], errors="ignore")
        assert a.id.is_unique and m.id.is_unique
        d = a.merge(m[["id", "author", "date"]], on="id", validate="one_to_one")
        d["year"] = pd.to_numeric(d.date)
        d["score"] = pd.to_numeric(d.ING_38_Role)
        d = d.dropna(subset=["year", "score"]).copy()
        d["author"] = d.author.replace(
            ["Auteur inconnu", "Unknown Author"], np.nan
        ).fillna("unknown_" + d.id)
        data[key] = d[["id", "year", "author", "score"]].copy()
    for key in ["babel_fr", "babel_uk"]:
        p = ROOT / f"results/latent/data/{key}_ING_38.csv"
        sources.append(data_source.file_source(p, ROOT))
        data[key] = pd.read_csv(p)[["id", "year", "author", "score"]]
    for key, d in data.items():
        assert d.id.is_unique and d.score.isin(range(7)).all()
        d["year"] = d.year.astype(int)
        d["score"] = d.score.astype(int)
        d.to_csv(OUT / f"{key}_data.csv", index=False)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    s["path"]: s.get("sha256", s.get("revision")) for s in sources
                },
                "bootstrap_repetitions": B,
                "bootstrap_seed": 42,
                "bootstrap_unit": "author groups, record weighted within each draw",
                "outcome": "ING_38",
            },
            indent=2,
        )
    )
    return data


def stats(d, t):
    h = d.score >= t
    n = len(d)
    nplus = int(h.sum())
    p = h.mean() if n else np.nan
    m = d.loc[h, "score"].mean()
    return {"n": n, "n_positive": nplus, "p": p, "m": m, "overall_mean": d.score.mean()}


def boot_stats(d, windows, t, rng):
    codes, groups = pd.factorize(d.author, sort=True)
    g = len(groups)
    A = np.zeros((g, len(windows) * 4))
    for j, (lo, hi) in enumerate(windows):
        hit = d.year.between(lo, hi).values
        positive = (d.score >= t).values & hit
        for k, weights in enumerate(
            [hit, positive, d.score.values * positive, d.score.values * hit]
        ):
            A[:, j * 4 + k] = np.bincount(codes, weights=weights, minlength=g)
    W = rng.multinomial(g, np.full(g, 1 / g), size=B)
    v = (W @ A).reshape(B, len(windows), 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack(
            [v[:, :, 1] / v[:, :, 0], v[:, :, 2] / v[:, :, 1], v[:, :, 3] / v[:, :, 0]],
            axis=-1,
        )


def main():
    data = load_data()
    periods = []
    contrastrows = []
    windowsrows = []
    rng = np.random.default_rng(42)
    for key, d in data.items():
        variants = {"all": d}
        if key == "roland_fr":
            lengths = pd.read_csv(ROOT / "results/length/linked_data.csv")[
                ["id", "n_chapters"]
            ]
            d = d.merge(lengths, on="id", validate="one_to_one")
            data[key] = d
            d.to_csv(OUT / f"{key}_data.csv", index=False)
            variants = {"all": d, "length_12_32": d[d.n_chapters.between(12, 32)]}
        for variant, z in variants.items():
            bins = [(v, v + 19) for v in sorted((z.year // 20 * 20).unique())]
            # Book comparisons focus on the latest decline, plus earlier century contrasts.
            if key == "theatre_uk":
                windows = [(1600, 1699), (1700, 1799)]
            elif key == "theatre_fr":
                windows = [(1600, 1699), (1700, 1799), (1800, 1899)]
            else:
                windows = [
                    (1700, 1799),
                    (1800, 1899),
                    (1900, min(1996, int(z.year.max()))),
                ]
            for t in [1, 3]:
                bp = boot_stats(z, bins, t, rng)
                for j, (lo, hi) in enumerate(bins):
                    s = stats(z[z.year.between(lo, hi)], t)
                    row = {
                        "corpus": key,
                        "variant": variant,
                        "threshold": t,
                        "start": lo,
                        "end": hi,
                        "year": z.loc[z.year.between(lo, hi), "year"].mean(),
                        **s,
                    }
                    for k, name in enumerate(["p", "m", "overall_mean"]):
                        row.update(
                            {
                                name + "_low": np.nanquantile(bp[:, j, k], 0.025),
                                name + "_high": np.nanquantile(bp[:, j, k], 0.975),
                            }
                        )
                    if t == 1 and s["n_positive"] > 0:
                        assert abs(s["overall_mean"] - s["p"] * s["m"]) < 1e-12
                    periods.append(row)
                b = boot_stats(z, windows, t, rng)
                point = []
                for j, (lo, hi) in enumerate(windows):
                    s = stats(z[z.year.between(lo, hi)], t)
                    point.append(s)
                    windowsrows.append(
                        {
                            "corpus": key,
                            "variant": variant,
                            "threshold": t,
                            "start": lo,
                            "end": hi,
                            **s,
                        }
                    )
                for i, j in [
                    (i, j)
                    for i in range(len(windows))
                    for j in range(i + 1, len(windows))
                ]:
                    s0, s1 = point[i], point[j]
                    delta = b[:, j, :] - b[:, i, :]
                    row = {
                        "corpus": key,
                        "variant": variant,
                        "threshold": t,
                        "early": f"{windows[i][0]}–{windows[i][1]}",
                        "late": f"{windows[j][0]}–{windows[j][1]}",
                        "n_early": s0["n"],
                        "n_late": s1["n"],
                        "n_positive_early": s0["n_positive"],
                        "n_positive_late": s1["n_positive"],
                    }
                    for k, name in enumerate(["p", "m", "overall_mean"]):
                        row.update(
                            {
                                name + "_early": s0[name],
                                name + "_late": s1[name],
                                name + "_change": s1[name] - s0[name],
                                name + "_change_low": np.nanquantile(
                                    delta[:, k], 0.025
                                ),
                                name + "_change_high": np.nanquantile(
                                    delta[:, k], 0.975
                                ),
                            }
                        )
                    if t == 1:
                        A = (s1["p"] - s0["p"]) * (s0["m"] + s1["m"]) / 2
                        Bv = (s1["m"] - s0["m"]) * (s0["p"] + s1["p"]) / 2
                        Ab = (b[:, j, 0] - b[:, i, 0]) * (b[:, j, 1] + b[:, i, 1]) / 2
                        Bb = (b[:, j, 1] - b[:, i, 1]) * (b[:, j, 0] + b[:, i, 0]) / 2
                        assert abs(A + Bv - row["overall_mean_change"]) < 1e-12
                        np.testing.assert_allclose(Ab + Bb, delta[:, 2], atol=1e-12)
                        row.update(
                            {
                                "prevalence_contribution": A,
                                "importance_contribution": Bv,
                                "prevalence_contribution_low": np.nanquantile(
                                    Ab, 0.025
                                ),
                                "prevalence_contribution_high": np.nanquantile(
                                    Ab, 0.975
                                ),
                                "importance_contribution_low": np.nanquantile(
                                    Bb, 0.025
                                ),
                                "importance_contribution_high": np.nanquantile(
                                    Bb, 0.975
                                ),
                                "prevalence_share": A / (A + Bv) if A + Bv else np.nan,
                            }
                        )
                    contrastrows.append(row)
    pd.DataFrame(periods).to_csv(OUT / "observed_periods.csv", index=False)
    pd.DataFrame(windowsrows).to_csv(OUT / "observed_centuries.csv", index=False)
    pd.DataFrame(contrastrows).to_csv(OUT / "observed_contrasts.csv", index=False)
    print(
        pd.DataFrame(contrastrows)
        .query('threshold==1 and variant=="all"')[
            [
                "corpus",
                "early",
                "late",
                "p_early",
                "p_late",
                "m_early",
                "m_late",
                "p_change_low",
                "p_change_high",
                "m_change_low",
                "m_change_high",
                "prevalence_share",
            ]
        ]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
