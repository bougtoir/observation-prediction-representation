PY ?= python3

all: results docs

# Everything that the public code repository can regenerate on its own.
results: data sims real canonical tables figs

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
	mkdir -p submission_package_no_supplement
	$(PY) scripts/build_docx.py

# QC + final SiM package: tests (junit), provenance, docx scans, zip.
package: docs
	$(PY) -m pytest tests/ -q --junitxml=analysis/pytest_report.xml
	$(PY) scripts/consistency_check.py
	$(PY) scripts/docx_residue_scan.py
	$(PY) scripts/pronoun_audit.py
	$(PY) scripts/finalize_package.py
	$(PY) scripts/clean_reproduction.py public
	$(PY) scripts/clean_reproduction.py source
	$(PY) scripts/finalize_package.py

test:
	$(PY) -m pytest tests/ -q

# CI mode for environments where the raw clinical dataset is unavailable:
# real-data tests are skipped explicitly; everything else must pass.
ci:
	OPR_ALLOW_DATA_SKIP=1 $(PY) -m pytest tests/ -q

# Full reproduction including data acquisition and the real-data tests.
full: all test
