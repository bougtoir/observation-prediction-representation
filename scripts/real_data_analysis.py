"""Real medical application: 30-day readmission, Diabetes 130-US
hospitals dataset (UCI 296).

Predefined comparison:
  M1  logistic regression on utilization/demographic predictors
  M2  rank-preserving monotone distortion g(s)=s^k/(s^k+(1-s)^k), k=1.6,
      of M1's predicted probabilities: identical ROC/AUC, distorted scale
      (k=1.6 keeps the distorted scale spread across ~[0,0.35])
  M3  Platt recalibration of M2 on the training fold

Evaluation on a held-out test fold shows what the OPR reveals that
AUROC alone conceals: M1 vs M2 share discrimination yet differ
completely in calibration and net benefit; M3 restores M1's scale.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from oprep import EmpiricalOPR, g_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG_SEED = 20261006

df = pd.read_csv(os.path.join(ROOT, "data/raw/diabetic_data.csv"), na_values="?")

# --- cohort: first encounter per patient (dedupe), binary outcome ---
df = df.sort_values("encounter_id").drop_duplicates("patient_nbr")
y = (df["readmitted"] == "<30").astype(int).values

age_map = {a: i for i, a in enumerate(
    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
     "[60-70)", "[70-80)", "[80-90)", "[90-100)"])}
feat = pd.DataFrame({
    "age": df["age"].map(age_map).fillna(4),
    "time_in_hospital": df["time_in_hospital"],
    "num_lab_procedures": df["num_lab_procedures"],
    "num_procedures": df["num_procedures"],
    "num_medications": df["num_medications"],
    "number_outpatient": df["number_outpatient"],
    "number_emergency": df["number_emergency"],
    "number_inpatient": df["number_inpatient"],
    "number_diagnoses": df["number_diagnoses"],
    "discharge_disposition_id": df["discharge_disposition_id"],
    "admission_type_id": df["admission_type_id"],
    "admission_source_id": df["admission_source_id"],
}).fillna(0).values

X_tr, X_te, y_tr, y_te = train_test_split(
    feat, y, test_size=0.30, random_state=RNG_SEED, stratify=y)

lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(X_tr, y_tr)
p1 = lr.predict_proba(X_te)[:, 1]
p1_tr = lr.predict_proba(X_tr)[:, 1]

K_DIST = 1.6
p2 = g_transform(p1, K_DIST)

# Platt recalibration of distorted scores on training fold
p2_tr = g_transform(p1_tr, K_DIST)
recal = LogisticRegression(max_iter=1000)
recal.fit(p2_tr.reshape(-1, 1), y_tr)
p3 = recal.predict_proba(p2.reshape(-1, 1))[:, 1]

models = {"M1_logistic": p1, "M2_distorted": p2, "M3_recalibrated": p3}

rows = []
PT = [0.02, 0.05, 0.10, 0.20]
for name, s in models.items():
    e = EmpiricalOPR(y_te, s)
    row = dict(model=name, n_test=len(y_te), prevalence=y_te.mean(),
               auc=e.auc(), brier=e.brier(), logloss=e.logloss())
    # calibration-in-the-large and slope (logit refit)
    z = np.clip(s, 1e-9, 1 - 1e-9)
    logit_s = np.log(z / (1 - z))
    X1 = np.c_[np.ones_like(logit_s), logit_s]
    import statsmodels.api as sm
    fit = sm.Logit(y_te, X1).fit(disp=0)
    row["calib_intercept"] = fit.params[0]
    row["calib_slope"] = fit.params[1]
    # integrated calibration index: E|m(s)-s| weighted by f(s), KDE est.
    grid = np.linspace(0.005, 0.995, 199)
    f0, f1 = EmpiricalOPR(y_te, s, method="kde").density(grid)
    fs = e.pi * f1 + (1 - e.pi) * f0
    sup = np.quantile(s, 0.995)
    m = e.calib_m(grid, bandwidth=max(0.005, 0.15 * sup))
    ici = np.nansum(np.abs(m - grid) * fs) / np.nansum(fs)
    row["ICI"] = ici
    for t in [0.05, 0.10, 0.20]:
        c = e.confusion(t)
        row[f"sens@{t}"] = c["sensitivity"]; row[f"spec@{t}"] = c["specificity"]
        row[f"ppv@{t}"] = c["PPV"]
    for pt in PT:
        row[f"nb@{pt}"] = e.net_benefit(pt)
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "analysis/realdata_metrics.csv"), index=False)
np.savez(os.path.join(ROOT, "data/processed/realdata_scores.npz"),
         y=y_te, **{f"s_{k}": v for k, v in models.items()})
print(out.to_string())
print("cohort n:", len(df), "events:", int(y.sum()),
      "prevalence:", round(y.mean(), 4))
