import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/romantic_length_mpl")
import matplotlib.style.core  # noqa: F401
import arviz as az
import numpy as np, pandas as pd
from pathlib import Path
from scipy.special import expit

OUT = Path(__file__).resolve().parents[1] / "results/length"
SRC = OUT.parent / "latent"
rows = []
coverage = []
for corpus in ["babel_fr", "babel_uk"]:
    d = pd.read_csv(SRC / f"data/{corpus}_ING_38.csv")
    post = az.from_netcdf(SRC / f"models/{corpus}_ING_38_quadratic.nc").posterior.stack(
        sample=("chain", "draw")
    )
    print(corpus, len(d), d.year.min(), d.year.max(), list(post.data_vars))
    t = post["threshold"].transpose("sample", ...).values
    years = np.array([1800, 1900, 1996])
    x = (years - d.year.mean()) / 100
    eta = (
        post["centered_year"].values[:, None] * x
        + post["I(centered_year ** 2)"].values[:, None] * x * x
    )
    scores = expit(eta[:, :, None] - t[:, None, :]).sum(2)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        v = scores[:, j] - scores[:, i]
        rows.append(
            {
                "corpus": corpus,
                "start": int(years[i]),
                "end": int(years[j]),
                "mean_change": float(v.mean()),
                "low": float(np.quantile(v, 0.025)),
                "high": float(np.quantile(v, 0.975)),
                "prob_decline": float((v < 0).mean()),
            }
        )
    for lo, hi in [(1500, 1619), (1620, 1810), (1811, 1899), (1900, 1996)]:
        coverage.append(
            {
                "corpus": corpus,
                "start": lo,
                "end": hi,
                "n": int(d.year.between(lo, hi).sum()),
            }
        )
    peak = d.year.mean() - 100 * post["centered_year"].values / (
        2 * post["I(centered_year ** 2)"].values
    )
    print("peak", np.quantile(peak, [0.025, 0.5, 0.975]))
pd.DataFrame(rows).to_csv(OUT / "babel_contrasts.csv", index=False)
pd.DataFrame(coverage).to_csv(OUT / "babel_coverage.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(coverage).to_string(index=False))
