# Contributing

Please open an issue describing the intended change before a substantial methodological revision. For errors, include the repository commit, operating system, Python/R versions, exact command, and a small example. Remove API keys and private text from logs.

Keep frozen study inputs unchanged unless a correction is explicitly documented. Put exploratory outputs in a separate directory. Changes to data, formulas, priors, or model settings require refitting affected models; saved files are not automatically invalidated.

Before proposing code changes:

```sh
make test
make check
```

For analysis changes, also run the affected stages and regenerate their table/figure exports. Explain any changes to sample counts or numerical results. For documentation edits, check relative links and run commands you changed where practical.

The GitHub Actions workflow runs the small offline checks and example on Linux. It does not refit the study or call paid annotation APIs. A full `make reproduce` run remains a separate local check.

Do not commit credentials, local environments, caches, or new paid annotation runs. Keep scientific changes separate from presentation changes so reviewers can assess their effects.
