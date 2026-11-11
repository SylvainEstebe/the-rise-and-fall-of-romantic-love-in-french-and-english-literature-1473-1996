# Releases

| Release | Contents | Verification and restoration |
| :--- | :--- | :--- |
| v0.2.0 | Current spline code, numerical results, figures, 58 fitted-model files and 48 predictive-draw arrays | `results/splines/binary_manifest.csv`, `binary_archives.json`; `script/restore_splines.py` |
| v0.1.0 | Earlier quadratic and decomposition analyses, 44 posterior files | `results/model_manifest.csv`, `model_archives.json`; `script/restore_models.py` |

Code, inputs, tables and figures are stored in Git. Large binaries are release ZIP assets with repository-relative paths and SHA-256 manifests.

`script/package_release.py` builds a source tarball. `--include-models` adds the restored v0.1.0 models; spline binaries have separate manifests.
