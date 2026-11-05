#!/usr/bin/env python3
"""Human validation, preserving the original blank-to-zero correction and work bootstrap."""

import csv
import glob
import json
import pathlib
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results/validation"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = {"ING_38": "ING_38 Romantic affiliation", "ING_40": "ING_40 Long-term love"}
ALL_KEYS = {
    **KEYS,
    "TH_01": "TH_01 Parental care",
    "TH_02": "TH_02 Parental investment",
    "TH_03": "TH_03 Romantic signaling",
}
LBL = {
    "ING_38": "Romantic affiliation",
    "ING_40": "Long-term love",
    "TH_01": "Parental care",
    "TH_02": "Parental investment",
    "TH_03": "Romantic signalling",
    "ING_41": "Obsessive thinking",
    "ING_42": "Romantic kiss",
    "ING_43": "Kiss (depicted)",
}
STRIP = re.compile(r"_part_\d+$")
RNG_SEED, B = 42, 2000


def load_human(coder, restore_zero=False):
    out = {}
    for f in glob.glob(str(ROOT / f"data/validation/{coder}/batch*")):
        for r in json.load(open(f)):
            ha = r.get("humanAnnotations") or {}
            rec = {}
            for c, k in KEYS.items():
                v = (ha.get(k) or {}).get("Role")
                v = None if v in (None, "", "null") else int(v)
                if v is None and restore_zero:
                    v = 0
                rec[c] = v
            out[STRIP.sub("", r["id"])] = rec
    return out


def load_gpt():
    out = {}
    for f in glob.glob(str(ROOT / "data/validation/gpt/batch_*.json")):
        for r in json.load(open(f)):
            out[STRIP.sub("", r["id"])] = {c: r.get(f"{c}_Role") for c in KEYS}
    return out


def pair(A, B):
    a, b = [], []
    per = {}
    ids = sorted(set(A) & set(B))
    for c in KEYS:
        x = [
            (int(A[i][c]), int(B[i][c]))
            for i in ids
            if A[i].get(c) is not None and B[i].get(c) is not None
        ]
        per[c] = x
        a += [p[0] for p in x]
        b += [p[1] for p in x]
    a, b = np.array(a), np.array(b)
    k = cohen_kappa_score(a, b, weights="quadratic", labels=list(range(7)))
    return dict(
        n=len(a),
        kappa_quadratic=k,
        spearman=spearmanr(a, b).correlation,
        per={
            c: (
                len(x),
                cohen_kappa_score(
                    [p[0] for p in x],
                    [p[1] for p in x],
                    weights="quadratic",
                    labels=list(range(7)),
                ),
            )
            for c, x in per.items()
        },
    )


def load_human_all(coder):
    """All five constructs for a coder (no zero-restore — arman was unaffected)."""
    out = {}
    for f in glob.glob(str(ROOT / f"data/validation/{coder}/batch*")):
        for r in json.load(open(f)):
            ha = r.get("humanAnnotations") or {}
            rec = {}
            for c, k in ALL_KEYS.items():
                v = (ha.get(k) or {}).get("Role")
                rec[c] = None if v in (None, "", "null") else int(v)
            out[STRIP.sub("", r["id"])] = rec
    return out


def load_gpt_all():
    out = {}
    for f in glob.glob(str(ROOT / "data/validation/gpt/batch_*.json")):
        for r in json.load(open(f)):
            out[STRIP.sub("", r["id"])] = {c: r.get(f"{c}_Role") for c in ALL_KEYS}
    return out


def pairs_by_work(A, B_, keys):
    """dict work_id -> [(a, b), ...] over the requested constructs (complete pairs only)."""
    out = {}
    for i in sorted(set(A) & set(B_)):
        x = [
            (int(A[i][c]), int(B_[i][c]))
            for c in keys
            if A[i].get(c) is not None and B_[i].get(c) is not None
        ]
        if x:
            out[i] = x
    return out


def point_and_ci(pw):
    """kappa_q + rho with 95% percentile bootstrap CIs, cluster-resampling works."""
    a = np.array([p[0] for x in pw.values() for p in x])
    b = np.array([p[1] for x in pw.values() for p in x])
    kq = cohen_kappa_score(a, b, weights="quadratic", labels=list(range(7)))
    rho = spearmanr(a, b).correlation
    rng = np.random.default_rng(RNG_SEED)
    ids = sorted(pw)
    kqs, rhos = [], []
    for _ in range(B):
        take = rng.integers(0, len(ids), len(ids))
        aa = [p[0] for j in take for p in pw[ids[j]]]
        bb = [p[1] for j in take for p in pw[ids[j]]]
        kqs.append(
            cohen_kappa_score(aa, bb, weights="quadratic", labels=list(range(7)))
        )
        rhos.append(spearmanr(aa, bb).correlation)
    kqs = np.array([k for k in kqs if np.isfinite(k)])
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    return dict(
        n=len(a),
        works=len(ids),
        kq=kq,
        kq_lo=np.percentile(kqs, 2.5),
        kq_hi=np.percentile(kqs, 97.5),
        rho=rho,
        rho_lo=np.percentile(rhos, 2.5),
        rho_hi=np.percentile(rhos, 97.5),
        dropped=2 * B - len(kqs) - len(rhos),
    )


def main():
    arm = load_human("arman")
    # The original interface failed to record zero scores for these two constructs.
    syl = load_human("sylvain", restore_zero=True)
    gpt = load_gpt()
    hm = {
        i: {
            c: int(round(np.mean([arm[i][c], syl[i][c]])))
            for c in KEYS
            if arm[i][c] is not None and syl[i][c] is not None
        }
        for i in sorted(set(arm) & set(syl))
    }
    rows = []
    for name, first, second, keys in [
        ("arman vs sylvain (human-human)", arm, syl, list(KEYS)),
        ("humans_mean vs GPT", hm, gpt, list(KEYS)),
        ("arman vs GPT", load_human_all("arman"), load_gpt_all(), list(ALL_KEYS)),
        ("sylvain vs GPT", syl, gpt, list(KEYS)),
    ]:
        for construct, selected in [("pooled", keys)] + [(k, [k]) for k in keys]:
            result = point_and_ci(pairs_by_work(first, second, selected))
            rows.append(
                dict(
                    comparison=name,
                    construct=construct,
                    n=result["n"],
                    works=result["works"],
                    kappa_q=result["kq"],
                    kappa_lo=result["kq_lo"],
                    kappa_hi=result["kq_hi"],
                    spearman=result["rho"],
                    spearman_lo=result["rho_lo"],
                    spearman_hi=result["rho_hi"],
                )
            )
    with (OUT / "validation_ci.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} validation comparisons saved to {OUT}")


if __name__ == "__main__":
    main()
