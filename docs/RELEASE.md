# Release files

[Home](../README.md) · [Spline reproduction](SPLINES.md)

The repository is private. Access to its releases follows the repository permissions.

| Release | Contents | Verification and restoration |
| :--- | :--- | :--- |
| v0.2.0 | Current spline code, numerical results, figures, 58 fitted-model files and 48 predictive-draw arrays | `results/splines/binary_manifest.csv`, `binary_archives.json`; `script/restore_splines.py` |
| v0.1.0 | Earlier quadratic and decomposition analyses, 44 posterior files | `results/model_manifest.csv`, `model_archives.json`; `script/restore_models.py` |

Large binary files are separate ZIP assets; each stores repository-relative paths. Code, frozen inputs, numerical tables and figures are in ordinary Git storage. GitHub's source ZIP contains the committed files; restoring the relevant binary archives supplies the saved samples. SHA-256 hashes are checked before extraction and for every restored file.

`script/package_release.py` builds a source tarball locally. Its optional `--include-models` flag applies to the historical v0.1.0 posterior set and requires those files to be restored first. It does not merge the old and spline model manifests. Use an empty output directory for each build.

Before publishing, run the offline tests, redraw spline figures, verify the numerical tables against the saved results, and inspect the eight submission figures. Full posterior verification and model fitting are distinct operations; record which was performed. Do not describe a figure-only check as a fresh estimation of the models.
