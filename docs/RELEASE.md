# Public release checklist

## Verify the release contents

- Include `README.md`, `CITATION.cff`, `LICENSE`, the guides, and the example.
- Include frozen annotation files, metadata, prompts, validation materials, and the primary text index.
- Include `analysis/submission_statistics.py` and its outputs in addition to the original analysis scripts.
- Run `make test`, `make check`, and `make reproduce`; inspect `results/reproduction.json`.
- Confirm every path in `results/article/tables.csv` and `figures.csv` exists.
- Verify the source-text redistribution inventory and preserve third-party attribution. The code license is not a blanket text license.
- Record the release commit and archive version. Add a publication DOI to the citation metadata only once assigned.

## Selected storage layout

The release uses ordinary Git storage for code, annotation inputs, metadata, tables, and figures. The 44 saved posterior files are distributed in eight separately downloadable ZIP assets. Their inventories are `results/model_manifest.csv` and `results/model_archives.json`. Restore downloaded ZIPs with `python3 script/restore_models.py --archives /path/to/downloads`.

The initial GitHub publication uses a new source-only history. The maintainer's original local history is preserved separately; no force push or destructive rewrite is required for the previously empty remote repository.

## Background: large model files

The original local project has approximately 7.3 GB of tracked results, including 28 files larger than 100 MiB. That original Git history contains these binaries. Adding an ignore rule alone would not remove them from history or make it suitable for an ordinary GitHub push.

For future migrations, two packaging options can retain the full research record:

1. Store posterior files in a versioned research archive or downloadable release assets, with a path/size/SHA-256 manifest. Keep code, frozen scores, metadata, tables, and figures in the regular repository. Restore each posterior to its original relative path before reusing saved draws.
2. Store posterior files through Git LFS, including the corresponding pointers and `.gitattributes`. Existing binary history needs a deliberate migration, and a fresh clone must retrieve the LFS objects.

Do not silently omit the posterior files while describing a release as including saved draws. If they are distributed separately, put the verified archive URL, version, checksums, and restoration command in the README. If they are intentionally unavailable, state that reproduction will fit missing models.

A storage migration is separate from reorganizing the working files. Preserve the existing history before any rewrite and verify a new clone after publication. Check the public links while signed out; an authenticated clone alone does not establish public accessibility.

## Prepare downloadable archives

After the complete reproduction run finishes:

```sh
python3 script/package_release.py --include-models
```

This writes `dist/release/romantic-love-source.tar.gz` and independently extractable model ZIPs, grouping at most 1 GiB of model input per ZIP. It also writes `model_manifest.csv` (original paths, sizes, SHA-256 hashes) and `archives.json` (archive sizes and hashes). The source archive excludes posterior binaries, credentials, caches, new annotation runs, and Git history. The model ZIPs preserve the original `results/...` paths. Extract all archives into the same directory to reconstruct the release.

Without `--include-models`, only the source archive and model manifest are built. Use an empty output directory for each build. Building is local: it does not publish files, create a release, or rewrite the current repository. Keep `dist/` out of Git.

## Known provenance gaps

The [data guide](../data/README.md) and [reproduction guide](REPRODUCIBILITY.md) document what is retained and what is not. A release should not imply that unretained acquisition scripts, original sampling logs, API responses, or human annotation interface logs are present.
