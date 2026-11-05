UV ?= uv
PY ?= $(UV) run --locked python
export MPLBACKEND := Agg
export MPLCONFIGDIR := /tmp/romantic_love_mpl
export PYTENSOR_FLAGS := compiledir=/tmp/romantic_love_pytensor
export R_LIBS_USER := $(CURDIR)/.Rlib
.DEFAULT_GOAL := help
.PHONY: help setup reproduce refit tables submission check test example prepare validation twolines descriptive figures primary signaling latent length two-part annotate-fr annotate-en annotate-theatre-fr annotate-theatre-uk annotate-babel-fr annotate-babel-uk annotate-validation
help:
	@echo 'setup | reproduce | refit | tables | check | prepare | validation | twolines | descriptive | figures'
	@echo 'Getting started: example (offline) | test | submission'
	@echo 'Bayesian fits: primary | signaling | latent | length | two-part'
	@echo 'API annotation: annotate-fr | annotate-en | annotate-theatre-fr | annotate-theatre-uk | annotate-babel-fr | annotate-babel-uk | annotate-validation'
setup:
	$(UV) sync --locked
	Rscript --vanilla requirements.R
check:
	$(PY) check.py
test:
	$(PY) -m unittest discover -s tests -v
example:
	$(PY) examples/summarize_scores.py
submission:
	$(PY) analysis/submission_statistics.py
prepare:
	$(PY) analysis/prepare.py
validation:
	$(PY) analysis/validation.py
twolines:
	$(PY) analysis/twolines.py
descriptive:
	$(PY) analysis/descriptive.py
figures:
	$(PY) analysis/histograms.py
	$(PY) analysis/figures_courtship.py
primary:
	$(PY) analysis/four_corpora.py
signaling:
	$(PY) analysis/text_signaling.py --twolines-script analysis/twolines.R
latent:
	$(PY) analysis/latent_corpora.py --twolines-script analysis/twolines.R
length: prepare
	$(PY) analysis/length.py
	$(PY) analysis/babel_contrasts.py
two-part: descriptive
	$(PY) analysis/two_part.py
annotate-fr:
	@for batch in data/texts/roland_fr/batch*.json; do $(PY) script/run_annotations.py --input "$$batch" --ingredients prompts/themes.txt --model gpt-5.4-2026-03-05 --output-dir results/new_annotations/roland_fr || exit $$?; done
annotate-en:
	$(PY) script/annotate_dir_gpt.py --input data/texts/novels_uk --tradition English --metadata data/annotations/main/novels_uk/metadata_english_works.csv --out results/new_annotations/novels_uk.json
annotate-theatre-fr:
	$(PY) script/annotate_dir_gpt.py --input data/texts/theatre_fr --tradition theatre --metadata data/annotations/main/theatre_fr/metadata_theatre_fr.csv --out results/new_annotations/theatre_fr.json
annotate-theatre-uk:
	$(PY) script/annotate_dir_gpt.py --input data/texts/theatre_uk --tradition theatre --metadata data/annotations/main/theatre_uk/metadata_theatre_uk.csv --out results/new_annotations/theatre_uk.json
annotate-babel-fr:
	$(PY) script/run_annotations_latent.py --input data/annotations/supplementary/babel_fr/metadata_babel_fr.csv --id-col id --title-col title --author-col author --date-col date --output-dir results/new_annotations/babel_fr
annotate-babel-uk:
	$(PY) script/run_batch_latent.py --input data/annotations/supplementary/babel_uk/metadata_babel_uk.csv --id-col id --title-col title --author-col author --date-col date --output-dir results/new_annotations/babel_uk

annotate-validation:
	$(PY) script/run_annotations.py --input data/validation/sample --ingredients prompts/themes.txt --model gpt-5.4-2026-03-05 --output-dir results/new_annotations/validation

reproduce:
	$(PY) reproduce.py
refit:
	$(PY) reproduce.py --refit
tables: submission
	$(PY) analysis/article_tables.py
