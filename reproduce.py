"""Recreate spline figures by default, or explicitly run the historical pipeline."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="Run the earlier quadratic/two-part pipeline; missing models may be fitted")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Resample all Bayesian models; otherwise reuse saved draws and fit missing models",
    )
    args = parser.parse_args()
    if args.refit and not args.legacy:
        parser.error("Use --legacy --refit for historical models; see docs/SPLINES.md for spline fitting commands.")
    if not args.legacy:
        subprocess.run([sys.executable, "analysis/splines/reproduce_figures.py"], cwd=ROOT, check=True)
        return
    if shutil.which("Rscript") is None:
        parser.error(
            "Install R (Rscript is required by the original two-lines analysis)."
        )
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    env.setdefault("PYTENSOR_FLAGS", "compiledir=" + str(ROOT / ".cache/pytensor"))
    env["R_LIBS_USER"] = str(ROOT / ".Rlib")
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        ["Rscript", "--vanilla", "requirements.R"], cwd=ROOT, env=env, check=True
    )
    steps = [
        ("check.py", False),
        ("analysis/prepare.py", False),
        ("analysis/validation.py", False),
        ("analysis/four_corpora.py", True),
        ("analysis/text_signaling.py", True),
        ("analysis/latent_corpora.py", True),
        ("analysis/twolines.py", False),
        ("analysis/length.py", True),
        ("analysis/babel_contrasts.py", False),
        ("analysis/descriptive.py", False),
        ("analysis/two_part.py", True),
        ("analysis/histograms.py", False),
        ("analysis/submission_statistics.py", False),
        ("analysis/article_tables.py", False),
    ]
    report = ROOT / "results/reproduction.json"
    report.parent.mkdir(exist_ok=True)
    runs = []
    for script, bayesian in steps:
        command = [sys.executable, script]
        if args.refit and bayesian:
            command.append("--refit")
        print(f"\n>>> {script}", flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=env)
        runs.append(
            dict(
                script=script,
                returncode=result.returncode,
                seconds=round(time.monotonic() - started, 2),
            )
        )
        report.write_text(
            json.dumps(dict(python=sys.version, refit=args.refit, steps=runs), indent=2)
            + "\n"
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    print("\nComplete: results/article/figures.csv and results/article/table_*.csv")


if __name__ == "__main__":
    main()
