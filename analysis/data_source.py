"""Load annotation tables from the local data/ folder or from Hugging Face.

When data/ is present it is used exactly as before. Otherwise scores, authors and
dates are read from the Hugging Face dataset. Only those columns are downloaded,
never the texts, and they are cached under .cache/hf/<revision>/.

Environment variables:
    ROMANTIC_LOVE_DATA          auto (default), local or hf
    ROMANTIC_LOVE_HF_REVISION   dataset branch, tag or commit (default: main)
    HF_TOKEN                    the dataset is gated; or run `hf auth login`

Check that both sources give the same tables (needs data/):
    uv run --locked python analysis/data_source.py --compare
"""

import argparse
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "SylvainEstebe/romantic_love_corpus"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}"
CACHE = ROOT / ".cache" / "hf"
ACCESS = (
    "The dataset is gated: accept its conditions on Hugging Face, then set HF_TOKEN "
    "or run `hf auth login`. To use a local copy, put it in data/ or set "
    "ROMANTIC_LOVE_DATA=local."
)

# corpus -> metadata file (text-based corpora)
MAIN = {
    "roland_fr": "metadata_roland_annotations.csv",
    "novels_uk": "metadata_english_works.csv",
    "theatre_fr": "metadata_theatre_fr.csv",
    "theatre_uk": "metadata_theatre_uk.csv",
}
# corpus -> (annotation file, metadata file) (knowledge-based Babel ratings)
BABEL = {
    "babel_fr": ("gpt/annotations_gpt.json", "metadata_babel_fr.csv"),
    "babel_uk": ("claude/annotations_claude.json", "metadata_babel_uk.csv"),
}
CONSTRUCTS = ("ING_38", "ING_40", "ING_41", "ING_42", "ING_43", "TH_01", "TH_02", "TH_03")
ROLES = [f"{code}_Role" for code in CONSTRUCTS]


class LocalDataMissing(FileNotFoundError):
    """A step needs files that exist only in the local data/ folder."""


def source(root=None):
    """Return "local" or "hf"."""
    choice = os.environ.get("ROMANTIC_LOVE_DATA", "auto").strip().lower()
    if choice not in {"auto", "local", "hf"}:
        raise ValueError("ROMANTIC_LOVE_DATA must be auto, local or hf")
    if choice == "auto":
        return "local" if ((root or ROOT) / "data" / "annotations").is_dir() else "hf"
    return choice


def require_local(*paths, root=None):
    root = root or ROOT
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        names = ", ".join(_relative(p, root) for p in missing)
        raise LocalDataMissing(
            f"Missing {names}. This step needs the local data/ folder, which is not "
            f"tracked in git and not published in {HF_URL}."
        )


def file_source(path, root=None):
    path = Path(path)
    return {
        "path": _relative(path, root or ROOT),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def tables(corpus, root=None):
    """Return (annotations, metadata, sources) for one corpus.

    annotations: one row per scored record, with id and <construct>_Role columns
    (Babel also has _known, a dict of familiarity flags per construct).
    metadata: id, title, author and the analysis date.
    sources: provenance entries, {"path", "sha256"} or {"path", "revision"}.
    IDs are left as stored so that callers can keep their own ID checks.
    """
    if corpus not in MAIN and corpus not in BABEL:
        raise KeyError(f"Unknown corpus {corpus!r}")
    if source(root) == "local":
        return _local_tables(corpus, root or ROOT)
    return _hf_tables(corpus)


def _relative(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _local_tables(corpus, root):
    folder = root / "data" / "annotations"
    if corpus in MAIN:
        paths = [
            folder / "main" / corpus / "annotations_gpt.json",
            folder / "main" / corpus / MAIN[corpus],
        ]
    else:
        annotation_file, metadata_file = BABEL[corpus]
        paths = [
            folder / "supplementary" / corpus / annotation_file,
            folder / "supplementary" / corpus / metadata_file,
        ]
    require_local(*paths, root=root)
    annotations = pd.DataFrame(json.loads(paths[0].read_text()))
    metadata = pd.read_csv(paths[1], dtype={"id": str})
    return annotations, metadata, [file_source(p, root) for p in paths]


@lru_cache(maxsize=None)
def hf_revision():
    """Resolve the requested dataset revision to a commit hash."""
    requested = os.environ.get("ROMANTIC_LOVE_HF_REVISION", "main").strip()
    if re.fullmatch(r"[0-9a-f]{40}", requested):
        return requested
    from huggingface_hub import HfApi

    try:
        return HfApi().dataset_info(HF_REPO, revision=requested).sha
    except Exception as error:
        raise RuntimeError(f"Could not open {HF_URL} at revision {requested!r}. {ACCESS}") from error


def _hf_frame(config, columns):
    """Read the requested columns of one config (the ones that exist), cached."""
    revision = hf_revision()
    key = hashlib.sha1(",".join(columns).encode()).hexdigest()[:8]
    cache = CACHE / revision / f"{config}-{key}.parquet"
    if cache.is_file():
        return pd.read_parquet(cache)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{HF_REPO}@{revision}/{config}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for {config} in {HF_URL}@{revision}")
    frames = []
    try:
        for path in files:
            # Read through a file handle: only the selected column chunks are fetched.
            with fs.open(path, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                keep = [c for c in columns if c in parquet.schema_arrow.names]
                frames.append(parquet.read(columns=keep).to_pandas())
    except Exception as error:
        if "gated" in str(error).lower() or "401" in str(error):
            raise RuntimeError(f"Could not read {config} from {HF_URL}. {ACCESS}") from error
        raise
    frame = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def _flag(value):
    return None if value is None or pd.isna(value) else bool(value)


def _hf_tables(corpus):
    if corpus in MAIN:
        columns = [
            "id", "work_id", "analysis_id", "analysis_date", "in_analysis",
            "title", "author", "date", *ROLES,
        ]  # fmt: skip
        frame = _hf_frame(corpus, columns)
        frame = frame.loc[frame["in_analysis"].fillna(False).astype(bool)].copy()
        if corpus == "roland_fr":
            # One row per source book; the scored passage has its own ID and date.
            frame["book_id"] = frame["id"]
            frame["id"] = frame["analysis_id"]
            frame["date"] = frame["analysis_date"]
        elif corpus == "novels_uk":
            # One row per part; scores and metadata are keyed by work.
            frame["id"] = frame["work_id"]
        if frame["id"].duplicated().any():
            raise ValueError(f"{corpus}: duplicate IDs among in_analysis rows in {HF_URL}")
        if "book_id" not in frame:
            frame["book_id"] = None  # as in the local files, set only for Roland
        roles = [c for c in ROLES if c in frame]
        annotations = frame[["id", "book_id", "date", *roles]]
        metadata = frame[["id", "title", "author", "date"]]
    else:
        columns = [
            "id", "book_id", "title", "author", "date", *ROLES,
            *(f"{code}_known" for code in CONSTRUCTS),
        ]  # fmt: skip
        frame = _hf_frame(corpus, columns)
        roles = [c for c in ROLES if c in frame]
        known = [c for c in frame if c.endswith("_known")]
        frame["_known"] = [
            {c.removesuffix("_known"): _flag(v) for c, v in zip(known, row)}
            for row in frame[known].itertuples(index=False, name=None)
        ]
        annotations = frame[["id", "book_id", "date", "title", "author", "_known", *roles]]
        metadata = frame[["id", "book_id", "title", "author", "date"]]
    sources = [{"path": f"hf://datasets/{HF_REPO}/{corpus}", "revision": hf_revision()}]
    return (
        annotations.reset_index(drop=True),
        metadata.reset_index(drop=True),
        sources,
    )


def _normalized(annotations, metadata):
    """One comparable row per ID: author and date from metadata, scores, flags."""
    roles = sorted(c for c in annotations if c.endswith("_Role"))
    merged = annotations.drop(columns=["date", "author", "title"], errors="ignore").merge(
        metadata[["id", "author", "date"]], on="id", how="left"
    )
    out = pd.DataFrame({"id": merged["id"].astype(str)})
    out["author"] = merged["author"].fillna("").astype(str).str.strip()
    date = merged["date"].replace(r"^\s*$", np.nan, regex=True)
    out["date"] = pd.to_numeric(date, errors="coerce")
    for column in roles:
        out[column] = pd.to_numeric(merged[column], errors="coerce")
    if "_known" in merged:
        out["_known"] = merged["_known"].map(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, dict) else ""
        )
    return out.set_index("id")


def compare(root=None):
    """Compare local and Hugging Face tables; return the number of differing corpora."""
    root = root or ROOT
    failures = 0
    for corpus in [*MAIN, *BABEL]:
        local = _normalized(*_local_tables(corpus, root)[:2])
        remote = _normalized(*_hf_tables(corpus)[:2])
        problems = []
        for label, table in [("local", local), ("hf", remote)]:
            if not table.index.is_unique:
                problems.append(f"duplicate IDs in {label}")
        if not problems:
            only_local = local.index.difference(remote.index)
            only_remote = remote.index.difference(local.index)
            if len(only_local):
                problems.append(f"{len(only_local)} IDs only local, e.g. {list(only_local[:3])}")
            if len(only_remote):
                problems.append(f"{len(only_remote)} IDs only on HF, e.g. {list(only_remote[:3])}")
            columns = sorted(set(local.columns) ^ set(remote.columns))
            if columns:
                problems.append(f"columns in one source only: {columns}")
            shared = local.index.intersection(remote.index)
            for column in [c for c in local.columns if c in remote.columns]:
                a, b = local.loc[shared, column], remote.loc[shared, column]
                differ = ~((a == b) | (a.isna() & b.isna()))
                if differ.any():
                    ids = list(differ[differ].index[:3])
                    problems.append(f"{column}: {int(differ.sum())} differ, e.g. {ids}")
        status = "OK" if not problems else "DIFFERENT"
        print(f"{corpus}: {status} (local {len(local)}, hf {len(remote)})")
        for problem in problems:
            print(f"  - {problem}")
        failures += bool(problems)
    print(f"Hugging Face revision: {hf_revision()}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--compare", action="store_true", help="Compare local data/ with Hugging Face"
    )
    args = parser.parse_args()
    if args.compare:
        raise SystemExit(1 if compare() else 0)
    current = source()
    print(f"Data source: {current}")
    if current == "hf":
        print(f"{HF_URL} at {hf_revision()}")


if __name__ == "__main__":
    main()
