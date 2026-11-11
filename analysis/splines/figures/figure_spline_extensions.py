"""Render the text and Babel spline figures once the new fits are complete."""

import argparse
from pathlib import Path

from figure_1_spline import CORPORA
from figure_1_spline import main as render

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/splines/figure_data/extensions"
SIGNALLING = ROOT / "results/splines/signalling"
BABEL = {"french_babel": "French Babel works", "english_babel": "English Babel works"}
SUPPLEMENT = ROOT / "results/splines/figures"


def main(verify_draws=False):
    render(verify_draws=verify_draws)
    render(
        verify_draws=verify_draws,
        corpora=CORPORA,
        sources={k: SIGNALLING / k for k in CORPORA},
        output=OUT / "signalling_text",
        figure_dir=SUPPLEMENT,
        filename="figure_romantic_signalling_spline",
        construct="Romantic signalling",
        code="TH_03",
    )
    render(
        verify_draws=verify_draws,
        corpora=BABEL,
        sources={
            "french_babel": ROOT / "results/splines/affiliation/french_babel",
            "english_babel": ROOT / "results/splines/affiliation_babel/english_babel",
        },
        output=OUT / "affiliation_babel",
        figure_dir=SUPPLEMENT,
        filename="figure_babel_affiliation_spline",
        mode="babel",
    )
    render(
        verify_draws=verify_draws,
        corpora=BABEL,
        sources={k: SIGNALLING / k for k in BABEL},
        output=OUT / "signalling_babel",
        figure_dir=SUPPLEMENT,
        filename="figure_babel_signalling_spline",
        mode="babel",
        construct="Romantic signalling",
        code="TH_03",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-draws", action="store_true")
    main(verify_draws=parser.parse_args().verify_draws)
