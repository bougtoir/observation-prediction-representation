PY ?= python3

all: data sims real canonical tables figs docs

data: data/raw/diabetic_data.csv
data/raw/diabetic_data.csv:
	bash scripts/acquire_data.sh

sims: data
	mkdir -p data/processed analysis
	$(PY) scripts/run_simulations.py

real: data
	mkdir -p data/processed analysis
	$(PY) scripts/real_data_analysis.py

canonical: sims real
	$(PY) scripts/make_canonical.py

tables: canonical
	mkdir -p tables
	$(PY) scripts/make_tables.py

figs: canonical
	mkdir -p figures
	$(PY) scripts/make_figures.py

docs: tables figs
	mkdir -p submission_package submission_package_kappa jce/submission_package_jce_kappa
	$(PY) scripts/build_docx.py
	$(PY) scripts/build_supplement.py
	$(PY) jce/build_jce.py

test:
	$(PY) -m pytest tests/ -q

# CI mode for environments where the raw clinical dataset is unavailable:
# real-data tests are skipped explicitly; everything else must pass.
ci:
	OPR_ALLOW_DATA_SKIP=1 $(PY) -m pytest tests/ -q

# Full reproduction including data acquisition and the real-data tests.
full: all test
