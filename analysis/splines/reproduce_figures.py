"""Redraw the current spline figures from frozen summaries, without fitting models."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-draws",
        action="store_true",
        help="Also audit plotted curves against the saved predictive draws",
    )
    args = parser.parse_args()
    commands = [
        ["analysis/splines/figures/figure_spline_extensions.py"]
        + (["--verify-draws"] if args.verify_draws else []),
        ["analysis/splines/figures/figure_babel_combined.py"],
        ["analysis/splines/publication_shape_supplement.py", "--stage", "figure"],
    ]
    report = {"fits_run": False, "verify_draws": args.verify_draws, "steps": []}
    for command in commands:
        started = time.monotonic()
        print("Running", " ".join(command), flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        report["steps"].append(
            {
                "command": command,
                "returncode": result.returncode,
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        (ROOT / "results/splines/reproduction.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    print("Spline figures regenerated. No models were fitted.")


if __name__ == "__main__":
    main()
