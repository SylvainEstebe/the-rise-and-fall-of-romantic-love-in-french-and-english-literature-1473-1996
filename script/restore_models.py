"""Verify and restore downloaded posterior archives using only the standard library."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid(path, expected):
    return path.is_file() and path.stat().st_size == int(expected["bytes"]) and digest(path) == expected["sha256"]


def restore_archive(archive_path, expected_files, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        seen = set()
        for info in archive.infolist():
            name = info.filename
            relative = PurePosixPath(name)
            target = destination.joinpath(*relative.parts)
            if relative.is_absolute() or ".." in relative.parts or not target.resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {name}")
            if name not in expected_files or name in seen:
                raise ValueError(f"Unexpected or duplicate archive entry: {name}")
            seen.add(name)
            expected = expected_files[name]
            if info.file_size != int(expected["bytes"]):
                raise ValueError(f"Unexpected size: {name}")
            if target.exists():
                if valid(target, expected):
                    continue
                raise ValueError(f"Existing file differs from the release: {target}. Move it aside before restoring.")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    h = hashlib.sha256()
                    with archive.open(info) as source:
                        while chunk := source.read(1024 * 1024):
                            h.update(chunk)
                            output.write(chunk)
                if h.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Checksum mismatch: {name}")
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archives", type=Path, help="Directory containing all eight downloaded model ZIPs")
    mode.add_argument("--check", action="store_true", help="Verify already restored models")
    args = parser.parse_args()
    with (ROOT / "results/model_manifest.csv").open() as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    if args.archives:
        archives = json.loads((ROOT / "results/model_archives.json").read_text())["archives"]
        for item in archives:
            archive = args.archives / item["file"]
            if not valid(archive, item):
                parser.error(f"Missing or damaged download: {archive}")
        for item in archives:
            print(f"Restoring {item['file']}", flush=True)
            restore_archive(args.archives / item["file"], expected, ROOT)
    failures = [name for name, item in expected.items() if not valid(ROOT / name, item)]
    if failures:
        parser.error("Missing or changed models: " + ", ".join(failures))
    print(f"Verified all {len(expected)} posterior files.")


if __name__ == "__main__":
    main()
