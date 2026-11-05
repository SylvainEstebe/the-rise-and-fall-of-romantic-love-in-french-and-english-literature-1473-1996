"""Reproduce the four two-lines tests and Holm correction over eight slopes."""

from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests
from four_corpora import CORPORA, load_corpus, run_twolines

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = pd.DataFrame(
        {
            name: run_twolines(load_corpus(name), ROOT / "analysis/twolines.R")
            for name in CORPORA
        }
    ).T
    columns = ["p_value_before_cutoff", "p_value_after_cutoff"]
    result[["p_before_holm", "p_after_holm"]] = multipletests(
        result[columns].to_numpy(float).ravel(), method="holm"
    )[1].reshape(-1, 2)
    out = ROOT / "results/primary/tables"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "twolines_with_holm.csv", index_label="corpus")


if __name__ == "__main__":
    main()
