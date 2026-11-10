# Observation–prediction representation for evaluating clinical prediction models

Code for the observation–prediction representation `R = {f0(s), f1(s), π}` of a
binary outcome and a continuous prediction score, its finite-sample estimator,
the Prediction Comb display, the ten-scenario simulation study, and the
application to the public Diabetes 130-US Hospitals dataset.

## Contents

- `scripts/oprep.py` — reusable estimator (density and empirical forms; ROC,
  calibration, confusion matrices, Cohen's κ, proper scores, net benefit as
  functionals of the representation)
- `scripts/prediction_comb.py` — Prediction Comb (binned joint mass on the original scale)
- `scripts/run_simulations.py` — simulation scenarios A–J (fixed seed)
- `scripts/acquire_data.sh` — downloads UCI dataset 296 and verifies SHA-256 checksums
- `scripts/real_data_analysis.py` — readmission models M1–M4 on the held-out fold
- `scripts/make_canonical.py`, `scripts/canonical.py` — canonical result registry
  (`analysis/canonical_results.json`) from which every reported number is read
- `scripts/make_tables.py`, `scripts/make_figures.py` — all tables and figures
- `tests/` — automated tests, including numerical-identity checks of the derived
  functionals against direct computation

## Reproduce

Python 3.10; exact package versions are pinned in `requirements.txt`.


```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make results PY=.venv/bin/python     # data → simulations → real data → registry → tables → figures
.venv/bin/python -m pytest tests/ -q
```

The real-data tests fail loudly when the dataset is missing; `make ci` skips
them explicitly (`OPR_ALLOW_DATA_SKIP=1`) for environments without network access.

## Data

Clore J, Cios K, DeShazo J, Strack B. Diabetes 130-US Hospitals for Years
1999–2008 [dataset]. UCI Machine Learning Repository; 2014.
doi:10.24432/C5230J (CC BY 4.0). The dataset is downloaded by
`scripts/acquire_data.sh`, not redistributed here.
