"""Build a source archive and optional model archives without rewriting Git history."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = [
    "README.md", "CITATION.cff", "CONTRIBUTING.md", "LICENSE", "Makefile",
    "pyproject.toml", "uv.lock", ".python-version", ".gitignore", ".gitattributes",
    ".env.example", "check.py", "reproduce.py", "requirements.R",
]
DIRECTORIES = [".github", "analysis", "data", "docs", "examples", "huggingface", "prompts", "results", "script", "src", "tests"]
SKIP_PARTS = {"__pycache__", ".DS_Store"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_files():
    paths = [ROOT / name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_symlink() or not path.is_file() or SKIP_PARTS.intersection(relative.parts):
                continue
            if relative.parts[:2] in [("results", "new_annotations"), ("results", "examples")]:
                continue
            if relative.as_posix() == "results/model_manifest.csv":
                continue  # A fresh manifest is inserted into every source archive.
            # Never collect local credential or temporary files from a directory.
            if path.name.startswith(".env") or path.suffix in {".pyc", ".tmp"}:
                continue
            paths.append(path)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/release")
    parser.add_argument("--include-models", action="store_true", help="Also package saved posterior files in ZIP archives of at most 1 GiB of input each")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory to avoid mixing releases.")
    output.mkdir(parents=True, exist_ok=True)
    files = release_files()
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        parser.error(f"Missing required release files: {missing}")
    models = [path for path in files if path.suffix == ".nc"]
    records = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in models]
    manifest = io.StringIO(newline="")
    writer = csv.DictWriter(manifest, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(records)
    manifest_bytes = manifest.getvalue().encode("utf-8")
    (output / "model_manifest.csv").write_bytes(manifest_bytes)
    source = output / "romantic-love-source.tar.gz"
    def portable_metadata(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(source, "w:gz") as archive:
        for path in files:
            if path.suffix != ".nc":
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False, filter=portable_metadata)
        entry = tarfile.TarInfo("results/model_manifest.csv")
        entry.size = len(manifest_bytes)
        archive.addfile(entry, io.BytesIO(manifest_bytes))
    archives = [source]
    if args.include_models:
        groups = []
        current, size = [], 0
        for path in models:
            if current and size + path.stat().st_size > 1024 ** 3:
                groups.append(current)
                current, size = [], 0
            current.append(path)
            size += path.stat().st_size
        if current:
            groups.append(current)
        for number, group in enumerate(groups, 1):
            destination = output / f"romantic-love-models-{number:02d}.zip"
            print(f"Packing {destination.name}", flush=True)
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
                for path in group:
                    archive.write(path, arcname=str(path.relative_to(ROOT)))
            archives.append(destination)
    index = [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in archives]
    (output / "archives.json").write_text(json.dumps({"models_included": args.include_models, "model_files": len(models), "archives": index}, indent=2) + "\n")
    print(f"Prepared {len(archives)} archives and a manifest for {len(models)} posterior files in {output}")


if __name__ == "__main__":
    main()
