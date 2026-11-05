"""Submission statistics from observed scores and saved posterior draws; no fitting."""

from pathlib import Path
import numpy as np
import pandas as pd
import h5netcdf

ROOT = Path(__file__).resolve().parents[1]
import four_corpora
import latent_corpora
import text_signaling


def main():
    OUT = ROOT / "results/submission"
    OUT.mkdir(exist_ok=True)
    labels = {
        "roland_fr": "French books (Roland)",
        "novels_uk": "English books (EarlyPrint–Gutenberg)",
        "theatre_fr": "French theatre",
        "theatre_uk": "English theatre",
        "babel_fr": "French Babel",
        "babel_uk": "English Babel",
    }
    descriptive = []
    turning = []
    for code in ["ING_38", "TH_03"]:
        for corpus, label in labels.items():
            if corpus.startswith("babel"):
                df = latent_corpora.load_corpus(f"{corpus}_{code}")
                model = ROOT / f"results/latent/models/{corpus}_{code}_quadratic.nc"
            elif code == "ING_38":
                df = four_corpora.load_corpus(corpus)
                model = ROOT / f"results/primary/models/{corpus}_quadratic.nc"
            else:
                df = text_signaling.load_corpus(ROOT, corpus)
                model = ROOT / f"results/signaling/models/{corpus}_{code}_quadratic.nc"
            x = df.score.to_numpy(dtype=float)
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], method="linear")
            descriptive.append(
                dict(
                    construct=code,
                    corpus=label,
                    n=len(x),
                    mean=x.mean(),
                    sd=x.std(ddof=1),
                    median=med,
                    q1=q1,
                    q3=q3,
                    iqr=q3 - q1,
                    n_positive=int((x > 0).sum()),
                    proportion_positive=(x > 0).mean(),
                    mean_publication_year=df.year.mean(),
                    year_min=df.year.min(),
                    year_max=df.year.max(),
                )
            )
            with h5netcdf.File(model, "r") as f:
                p = f.groups["posterior"]
                beta1 = np.asarray(p.variables["centered_year"][:]).reshape(-1)
                beta2 = np.asarray(p.variables["I(centered_year ** 2)"][:]).reshape(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                year = df.year.mean() + 100 * (-beta1 / (2 * beta2))
            finite = np.isfinite(year)
            assert finite.all(), (corpus, code, "nonfinite turning draws")
            low, median, high = np.quantile(year, [0.025, 0.5, 0.975])
            turning.append(
                dict(
                    construct=code,
                    corpus=label,
                    n=len(df),
                    posterior_draws=len(year),
                    year_median=median,
                    year_low=low,
                    year_high=high,
                    prob_concave=float((beta2 < 0).mean()),
                    prob_turn_in_observed_period=float(
                        ((year >= df.year.min()) & (year <= df.year.max())).mean()
                    ),
                    prob_peak_in_observed_period=float(
                        (
                            (beta2 < 0)
                            & (year >= df.year.min())
                            & (year <= df.year.max())
                        ).mean()
                    ),
                    beta2_mean=beta2.mean(),
                    model_path=str(model.relative_to(ROOT)),
                )
            )
    a = pd.DataFrame(descriptive)
    b = pd.DataFrame(turning)
    a.to_csv(OUT / "descriptive_statistics.csv", index=False)
    b.to_csv(OUT / "quadratic_turning_points.csv", index=False)
    pd.read_csv(ROOT / "results/primary/tables/twolines_with_holm.csv").to_csv(
        OUT / "two_lines.csv", index=False
    )
    v = pd.read_csv(ROOT / "results/validation/validation_ci.csv")
    selection = [
        ("arman vs sylvain (human-human)", "ING_38"),
        ("humans_mean vs GPT", "ING_38"),
        ("arman vs GPT", "TH_03"),
    ]
    v = pd.concat(
        [
            v[(v.comparison == comparison) & (v.construct == code)]
            for comparison, code in selection
        ]
    )
    v.to_csv(OUT / "validation.csv", index=False)
    print(
        "DESCRIPTIVES\n" + a.to_string(index=False, float_format=lambda x: f"{x:.12g}")
    )
    print(
        "TURNING POINTS\n"
        + b.drop(columns="model_path").to_string(
            index=False, float_format=lambda x: f"{x:.12g}"
        )
    )
    print("VALIDATION\n" + v.to_string(index=False, float_format=lambda x: f"{x:.15g}"))


if __name__ == "__main__":
    main()
