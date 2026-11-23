"""Bayesian ordinal trajectories of romantic affiliation in the four text corpora."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style.core  # noqa: F401  compatibilité ArviZ / Matplotlib 3.11, à importer avant ArviZ et Bambi
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

import arviz as az
import runtime  # noqa: F401
import bambi as bmb

import data_source

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "primary"
REFIT = False

# corpus -> fichier de métadonnées (auteur et date)
CORPORA = {
    "roland_fr": "metadata_roland_annotations.csv",
    "novels_uk": "metadata_english_works.csv",
    "theatre_fr": "metadata_theatre_fr.csv",
    "theatre_uk": "metadata_theatre_uk.csv",
}

PERIOD = 20
MIN_WORKS = 10
ALPHA = 0.05
Z = norm.ppf(0.975)

COLORS = {
    "observations": "#8A8A8A",
    "summaries": "#0072B2",
    "trend": "#D55E00",
}

TWOLINES_URL = "https://webstimate.org/twolines/twolines.R"
R_CODE = r"""
args <- commandArgs(trailingOnly=TRUE)
env <- new.env(parent=globalenv()); sys.source(args[1], envir=env)
res <- env$twolines(score ~ centered_years, data=read.csv(args[2]), graph=0)
write.csv(data.frame(slope_before_cutoff=res$b1, p_value_before_cutoff=res$p1, slope_after_cutoff=res$b2, p_value_after_cutoff=res$p2,
                     cutoff_x=as.numeric(res$xc)), args[3], row.names=FALSE)
"""

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    }
)

rng = np.random.default_rng(0)


# --- Données -----------------------------------------------------------------


def load_corpus(name):
    """Annotations GPT, avec auteur et date tirés des métadonnées (les annotations du théâtre n'ont pas de date)."""
    df, metadata, _ = data_source.tables(name)
    # Empty columns such as book_id stay float NaN, as pd.read_json returned them.
    df = df.drop(columns="date").apply(lambda s: s.astype(float) if s.isna().all() else s)
    df = df.merge(metadata[["id", "author", "date"]], on="id")
    df["year"] = pd.to_numeric(df["date"])
    df["score"] = pd.to_numeric(df["ING_38_Role"])
    df["centered_year"] = (df.year - df.year.mean()) / 100
    df["score_categories"] = pd.Categorical(df.score.astype(int), ordered=True)
    df = df.dropna(subset=["year", "score"])
    df["author"] = df.author.replace(
        ["Auteur inconnu", "Unknown Author"], np.nan
    ).fillna("unknown_" + df.id)
    return df


def period_means(df):
    """Moyenne et erreur standard du score par période, pour les périodes d'au moins MIN_WORKS œuvres."""
    return (
        df.groupby(PERIOD * (df.year // PERIOD))
        .agg(
            year=("year", "mean"),
            mean=("score", "mean"),
            sem=("score", "sem"),
            n=("score", "size"),
        )
        .query(f"n >= {MIN_WORKS}")
    )


# --- Modèles ordinaux --------------------------------------------------------


def fit_ordinal(formula, df, path):
    model = bmb.Model(formula, df, family="cumulative")
    if path.exists() and not REFIT:
        print(f"Loading {path.name}", flush=True)
        return model, az.from_netcdf(path)
    idata = model.fit(
        draws=4000,
        tune=2000,
        chains=4,
        cores=4,
        progressbar=False,
        random_seed=0,
        idata_kwargs={"log_likelihood": True},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(path)
    return model, idata


def predict_ordinal_trend(model, idata, df, levels):
    """Probabilités par catégorie et score attendu de l'auteur moyen, avec IC crédible à 95 %."""
    grid = np.linspace(df.year.min(), df.year.max(), 400)
    new = pd.DataFrame({"centered_year": (grid - df.year.mean()) / 100})
    prediction = model.predict(
        idata, data=new, include_group_specific=False, inplace=False
    ).posterior["p"]
    expected = (prediction * levels).sum(prediction.dims[-1])
    lower, upper = expected.quantile([0.025, 0.975], ("chain", "draw")).values
    return {
        "x": grid,
        "mean": expected.mean(("chain", "draw")).values,
        "lower": lower,
        "upper": upper,
        "probs": prediction.mean(("chain", "draw")).values,
    }


# --- Two lines ---------------------------------------------------------------


def download_twolines(folder):
    """Use the archived study implementation; no network dependency."""
    import shutil

    source = Path(__file__).resolve().parent / "twolines.R"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "twolines.R"
    shutil.copyfile(source, target)
    return target


def run_twolines(df, script):
    """Test two-lines de Simonsohn ; rise_fall si pente positive puis négative, toutes deux significatives."""
    mean_year = df.year.mean()
    with tempfile.TemporaryDirectory() as tmp:
        df.assign(centered_years=(df.year - mean_year) / 100)[
            ["score", "centered_years"]
        ].to_csv(f"{tmp}/in.csv", index=False)
        subprocess.run(
            [
                "Rscript",
                "--vanilla",
                "-e",
                R_CODE,
                str(script),
                f"{tmp}/in.csv",
                f"{tmp}/out.csv",
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        result = pd.read_csv(f"{tmp}/out.csv").iloc[0]

    res = {
        "n": len(df),
        "cutoff_year": mean_year + 100 * result.cutoff_x,
        **result.drop("cutoff_x"),
    }
    res["rise_fall"] = (
        res["slope_before_cutoff"] > 0
        and res["slope_after_cutoff"] < 0
        and max(res["p_value_before_cutoff"], res["p_value_after_cutoff"]) < ALPHA
    )
    return pd.Series(res)


# --- Processus gaussien ------------------------------------------------------


def fit_gaussian_process(periods):
    """Tendance (Matern 3/2) sur les moyennes par période de 20 ans : bruit de mesure connu (SEM) fixé via `alpha`,
    dispersion résiduelle entre périodes laissée à un WhiteKernel libre (sinon le Matern doit la mimer en devenant
    trop local)."""
    kernel = C(0.5, (1e-3, 1e3)) * Matern(
        length_scale=100, length_scale_bounds=(20, 1000), nu=1.5
    ) + WhiteKernel(0.1, (1e-3, 1e2))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=periods["sem"].to_numpy(float) ** 2,
        n_restarts_optimizer=20,
        random_state=0,
    )
    return model.fit(
        periods[["year"]].to_numpy(float),
        periods["mean"].to_numpy(float) - periods["mean"].mean(),
    )


def gaussian_process_trend(gp, df, periods):
    """Postérieur de la seule composante lisse (le WhiteKernel est traité comme du bruit, exclu de l'IC tracé)."""
    years = np.linspace(df.year.min(), df.year.max(), 1000)
    k_trend = gp.kernel_.k1
    K_s = k_trend(years[:, None], periods[["year"]].to_numpy(float))
    v = solve_triangular(gp.L_, K_s.T, lower=True)
    mean = K_s @ gp.alpha_ + periods["mean"].mean()
    sd = np.sqrt(np.diag(k_trend(years[:, None])) - (v**2).sum(0))
    return {"x": years, "mean": mean, "lower": mean - Z * sd, "upper": mean + Z * sd}


def kernel_hyperparameters(gp):
    params = gp.kernel_.get_params()
    return pd.Series({h.name: params[h.name] for h in gp.kernel_.hyperparameters})


# --- Tableaux ----------------------------------------------------------------


def flatten(frame):
    """Tableau lignes × statistiques -> série indexée « ligne.statistique »."""
    stacked = frame.stack(future_stack=True)
    stacked.index = [f"{row}.{stat}" for row, stat in stacked.index]
    return stacked


def loo_scalars(loo):
    return pd.Series({key: value for key, value in loo.items() if np.ndim(value) == 0})


def export_tables(results):
    for table in next(iter(results.values()))["tables"]:
        long = pd.concat(
            {name: r["tables"][table] for name, r in results.items()},
            names=["corpus", "parametre"],
        )
        long.rename("resultat").reset_index().to_csv(
            OUTPUT / "tables" / f"{table}.csv", index=False
        )


# --- Analyse d'un corpus -----------------------------------------------------


def analyse_corpus(name, twolines_script):
    print(f"Corpus {name}")
    df = load_corpus(name)
    levels = np.array(df.score_categories.cat.categories, dtype=float)
    _, idata_lin = fit_ordinal(
        "score_categories ~ centered_year + (1|author)",
        df,
        OUTPUT / "models" / f"{name}_linear.nc",
    )
    model, idata = fit_ordinal(
        "score_categories ~ centered_year + I(centered_year**2) + (1|author)",
        df,
        OUTPUT / "models" / f"{name}_quadratic.nc",
    )
    periods = period_means(df)
    gp = fit_gaussian_process(periods)
    return {
        "df": df,
        "levels": levels,
        "periods": periods,
        "ordinal": predict_ordinal_trend(model, idata, df, levels),
        "gp": gaussian_process_trend(gp, df, periods),
        "tables": {
            "ordinal_summary": flatten(
                az.summary(
                    idata,
                    var_names=[
                        "centered_year",
                        "I(centered_year ** 2)",
                        "1|author_sigma",
                    ],
                )
            ),
            "loo": loo_scalars(az.loo(idata)),
            "model_comparison": flatten(
                az.compare({"lineaire": idata_lin, "quadratique": idata})
            ),
            "twolines": run_twolines(df, twolines_script),
            "gp_kernel": kernel_hyperparameters(gp),
        },
    }


# --- Figures -----------------------------------------------------------------


def save(fig, filename):
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT / "figures" / f"{filename}.{ext}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_trend_grid(results, key, trend_label, title, filename):
    """Grille 2x2 : observations, moyennes par période et tendance ± IC 95 %, un corpus par panneau."""
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (name, r) in zip(axes.flat, results.items()):
        df, periods, trend, levels = r["df"], r["periods"], r[key], r["levels"]
        ax.scatter(
            df.year + rng.uniform(-1.5, 1.5, len(df)),
            df.score + rng.uniform(-0.18, 0.18, len(df)),
            s=5,
            color=COLORS["observations"],
            alpha=0.12,
            edgecolors="none",
            label="Individual observed scores",
        )
        ax.errorbar(
            periods.year,
            periods["mean"],
            yerr=Z * periods["sem"],
            fmt="o",
            markersize=4,
            color=COLORS["summaries"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=0.9,
            capsize=2,
            zorder=3,
            label=f"{PERIOD}-year mean ± {Z:.2f} SEM",
        )
        ax.plot(
            trend["x"],
            trend["mean"],
            color=COLORS["trend"],
            linewidth=2.3,
            zorder=4,
            label=trend_label,
        )
        ax.fill_between(
            trend["x"],
            trend["lower"],
            trend["upper"],
            color=COLORS["trend"],
            alpha=0.13,
            linewidth=0,
            zorder=2,
            label="95% credible interval",
        )
        ax.set(
            ylim=(levels.min() - 0.45, levels.max() + 0.45),
            yticks=levels,
            title=f"{name} (n = {len(df):,})",
        )
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    fig.supylabel(f"Romantic affiliation score ({levels.min():.0f}–{levels.max():.0f})")
    fig.suptitle(title)
    legend = fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2
    )
    legend.legend_handles[0].set_alpha(0.45)
    save(fig, filename)


def plot_probability_grid(results):
    """Un corpus par colonne : probabilités prédites par catégorie, scores observés et prédits, effectifs par période."""
    fig, axes = plt.subplots(
        3,
        len(results),
        figsize=(16, 10),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [3, 3, 1]},
        layout="constrained",
    )
    for column, (name, r) in zip(axes.T, results.items()):
        df, periods, trend, levels = r["df"], r["periods"], r["ordinal"], r["levels"]
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
        column[0].stackplot(
            trend["x"], trend["probs"].T, colors=colors, labels=levels.astype(int)
        )
        column[0].set_title(name)

        column[1].scatter(
            df.year + rng.uniform(-1.5, 1.5, len(df)),
            df.score + rng.uniform(-0.25, 0.25, len(df)),
            s=6,
            alpha=0.15,
            color="tab:blue",
            edgecolor="none",
            zorder=0,
            label="Observations",
        )
        column[1].plot(
            trend["x"],
            trend["mean"],
            color="tab:blue",
            label="expected score (bayésien, auteur moyen)",
        )
        column[1].fill_between(
            trend["x"],
            trend["lower"],
            trend["upper"],
            alpha=0.25,
            label="IC crédible 95 %",
        )
        column[1].errorbar(
            periods.year,
            periods["mean"],
            yerr=Z * periods["sem"],
            fmt="o",
            color="k",
            ms=4,
            label="mean score",
        )

        column[2].bar(periods.year, periods.n, width=20, color="grey")
    axes[0, 0].set_ylabel("expected probabilities")
    axes[1, 0].set_ylabel("Score")
    axes[2, 0].set_ylabel("n")
    axes[0, -1].legend(title="Score", loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[1, 0].legend()
    fig.supxlabel("Year")
    save(fig, "figure_ordinal_probabilities")


def main():
    global REFIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Resample instead of loading saved posterior draws",
    )
    REFIT = parser.parse_args().refit
    for folder in ("figures", "tables"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        twolines_script = download_twolines(Path(tmp))
        results = {}
        for name in CORPORA:
            results[name] = analyse_corpus(name, twolines_script)
            export_tables(results)
    export_tables(results)
    plot_trend_grid(
        results,
        "ordinal",
        "Model-predicted mean score",
        "Romantic affiliation over time",
        "figure_ordinal",
    )
    plot_trend_grid(
        results,
        "gp",
        "Gaussian process predicted mean score",
        "Romantic affiliation over time (Gaussian process regression)",
        "figure_gp",
    )
    plot_probability_grid(results)


if __name__ == "__main__":
    main()
