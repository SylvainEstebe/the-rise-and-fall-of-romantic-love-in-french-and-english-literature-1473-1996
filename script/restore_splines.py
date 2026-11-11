"""Download and verify the saved spline draws or fitted models from release v0.2.0."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import urllib.request
import urllib.error
from urllib.parse import urlsplit

try:
    from .restore_models import restore_archive, valid
except ImportError:  # Direct execution from the repository root.
    from restore_models import restore_archive, valid

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996/releases/download/v0.2.0/"


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    """Keep a private-repository token on GitHub’s API, not on asset-storage redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            redirected is not None
            and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def open_asset(filename):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    opener = urllib.request.build_opener(ReleaseRedirect())
    if not token:
        return opener.open(RELEASE + filename)
    api = "https://api.github.com/repos/SylvainEstebe/the-rise-and-fall-of-romantic-love-in-french-and-english-literature-1473-1996"
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
    }
    with opener.open(
        urllib.request.Request(api + "/releases/tags/v0.2.0", headers=headers)
    ) as response:
        release = json.load(response)
    asset = next((a for a in release["assets"] if a["name"] == filename), None)
    if asset is None:
        raise ValueError("Missing release asset: " + filename)
    headers["Accept"] = "application/octet-stream"
    return opener.open(urllib.request.Request(asset["url"], headers=headers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["draws", "models", "all"], default="draws")
    parser.add_argument(
        "--archives", type=Path, default=ROOT / "model-downloads/splines"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify restored files; do not download or extract",
    )
    args = parser.parse_args()
    index = json.loads((ROOT / "results/splines/binary_archives.json").read_text())
    archives = [
        a for a in index["archives"] if args.kind == "all" or a["kind"] == args.kind
    ]
    paths = {p for a in archives for p in a["paths"]}
    with (ROOT / "results/splines/binary_manifest.csv").open() as f:
        expected = {r["path"]: r for r in csv.DictReader(f) if r["path"] in paths}
    if not args.check:
        args.archives.mkdir(parents=True, exist_ok=True)
        for item in archives:
            if all(valid(ROOT / p, expected[p]) for p in item["paths"]):
                continue
            path = args.archives / item["file"]
            if not valid(path, item):
                partial = path.with_suffix(".partial")
                print("Downloading", item["file"], flush=True)
                try:
                    with (
                        open_asset(item["file"]) as src,
                        partial.open("wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    if not valid(partial, item):
                        raise ValueError("Archive checksum mismatch: " + item["file"])
                    partial.replace(path)
                except urllib.error.HTTPError as error:
                    parser.error(
                        f"Could not download {item['file']} (HTTP {error.code}). For a private repository, set GH_TOKEN/GITHUB_TOKEN with read access, or download the ZIPs in your browser and use --archives."
                    )
                finally:
                    partial.unlink(missing_ok=True)
            restore_archive(path, expected, ROOT)
    failed = [p for p, record in expected.items() if not valid(ROOT / p, record)]
    if failed:
        parser.error("Missing or changed files: " + ", ".join(failed))
    print(f"Verified {len(expected)} spline {args.kind} files.")


if __name__ == "__main__":
    main()
