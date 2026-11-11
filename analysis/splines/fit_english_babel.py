"""Complete the English Babel ING_38 spline suite using the final paper settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import ordinal_shape_robustness as a

OUT = ROOT / "results/splines/affiliation_babel"
if __name__ == "__main__":
    a.run_dataset(
        ROOT / "results/courtship/babel_uk_data.csv",
        "english_babel",
        OUT,
        draws=4000,
        tune=2000,
        chains=4,
        cores=4,
        seed=0,
        expected_n=826,
        expected_range=(1485, 2023),
    )
    a.exact_reloo_dataset(OUT / "english_babel", cores=4)
    a.summarize_dataset(OUT / "english_babel", OUT / "english_babel", "english_babel")
