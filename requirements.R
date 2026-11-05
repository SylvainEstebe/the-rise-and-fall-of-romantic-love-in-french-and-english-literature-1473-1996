pins <- c(mgcv="1.9-0", stringr="1.5.1", sandwich="3.1-0", lmtest="0.9-40")
lib <- Sys.getenv("R_LIBS_USER", unset=file.path(getwd(), ".Rlib"))
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
missing <- names(pins)[!vapply(names(pins), function(p) {
  requireNamespace(p, quietly=TRUE) && packageVersion(p) == package_version(pins[[p]])
}, logical(1))]
if (length(missing)) {
  if (!requireNamespace("remotes", quietly=TRUE))
    install.packages("remotes", lib=lib, repos="https://cloud.r-project.org")
  for (p in missing) remotes::install_version(p, version=pins[[p]], lib=lib,
    repos="https://cloud.r-project.org", upgrade="never")
}
