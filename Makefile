PY ?= python3

all: data sims real tables figs docs

data: data/raw/diabetic_data.csv
data/raw/diabetic_data.csv:
	@echo "download per data/raw/ACQUISITION_LEDGER.md" && test -f data/raw/diabetic_data.csv

sims: data
	$(PY) scripts/run_simulations.py

real: data
	$(PY) scripts/real_data_analysis.py

tables: sims real
	$(PY) scripts/make_tables.py

figs: sims real
	$(PY) scripts/make_figures.py

docs: tables figs
	$(PY) scripts/build_docx.py
	$(PY) scripts/build_supplement.py

test:
	$(PY) -m pytest tests/ -q
