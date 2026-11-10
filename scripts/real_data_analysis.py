"""Real medical application: 30-day readmission, Diabetes 130-US
hospitals dataset (UCI 296).

Predefined comparison:
  M1  logistic regression on utilization/demographic predictors
  M2  rank-preserving monotone distortion g(s)=s^k/(s^k+(1-s)^k), k=1.6,
      of M1's predicted probabilities: identical ROC/AUC, distorted scale
      (k=1.6 keeps the distorted scale spread across ~[0,0.35]).
      This is a designed demonstration of non-identifiability, not a
      naturally occurring clinical comparator.
  M3  Platt recalibration of M2 on the training fold
  M4  gradient-boosted trees on the same predictors: a genuinely
      different model class, prespecified as the natural comparison

Evaluation on a held-out test fold shows what the OPR reveals that
AUROC alone conceals: M1 vs M2 share discrimination yet differ
completely in calibration and net benefit; M3 restores M1's scale.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from oprep import EmpiricalOPR, g_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG_SEED = 20261006

df = pd.read_csv(os.path.join(ROOT, "data/raw/diabetic_data.csv"), na_values="?")
n_encounters = len(df)

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

# Standard logistic (Platt) recalibration of the distorted scores, fitted
# on the logit of p2 on the training fold only. Since g satisfies
# logit(g(s)) = k*logit(s), the recalibration slope should recover ~1/k.
p2_tr = g_transform(p1_tr, K_DIST)
EPS = 1e-9
logit2_tr = np.log(np.clip(p2_tr, EPS, 1 - EPS)
                   / np.clip(1 - p2_tr, EPS, 1 - EPS))
recal = LogisticRegression(max_iter=1000)
recal.fit(logit2_tr.reshape(-1, 1), y_tr)
logit2_te = np.log(np.clip(p2, EPS, 1 - EPS) / np.clip(1 - p2, EPS, 1 - EPS))
p3 = recal.predict_proba(logit2_te.reshape(-1, 1))[:, 1]
recal_a = float(recal.intercept_[0])
recal_b = float(recal.coef_[0, 0])
print(f"recalibration: logit(P) = {recal_a:.4f} + {recal_b:.4f}*logit(p2)"
      f"  (expected b ~ 1/k = {1.0 / K_DIST:.4f})")
pd.DataFrame([dict(intercept=recal_a, slope=recal_b, k=K_DIST,
                   expected_slope=1.0 / K_DIST)]).to_csv(
    os.path.join(ROOT, "analysis/recalibration_fit.csv"), index=False)

gbm = HistGradientBoostingClassifier(random_state=RNG_SEED)
gbm.fit(X_tr, y_tr)
p4 = gbm.predict_proba(X_te)[:, 1]

models = {"M1_logistic": p1, "M2_distorted": p2, "M3_recalibrated": p3,
          "M4_gbm": p4}

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

# --- Cohen's kappa: threshold-dependent agreement functional --------
# kappa(t) is computed only after the classification rule S>=t; it is
# agreement, not clinical utility, and is both threshold- and
# prevalence-dependent.
KTGRID = np.linspace(0.005, 0.30, 400)
kap_rows, curve_rows = [], {"t": KTGRID}
for name, s in models.items():
    e = EmpiricalOPR(y_te, s)
    ks = np.array([e.kappa(t) for t in KTGRID])
    curve_rows[name] = ks
    j = int(np.nanargmax(ks))
    row = dict(model=name,
               kappa_max=float(ks[j]), t_at_kappa_max=float(KTGRID[j]))
    for t in [0.02, 0.05, 0.10, 0.20]:
        row[f"kappa@{t}"] = e.kappa(t)
    # NB-optimal threshold for the agreement-vs-utility comparison
    nbs = np.array([e.net_benefit(t) for t in KTGRID])
    jnb = int(np.nanargmax(nbs))
    row["t_at_nb_max"] = float(KTGRID[jnb])
    kap_rows.append(row)
pd.DataFrame(kap_rows).to_csv(
    os.path.join(ROOT, "analysis/realdata_kappa.csv"), index=False)
pd.DataFrame(curve_rows).to_csv(
    os.path.join(ROOT, "analysis/kappa_curves.csv"), index=False)
print(pd.DataFrame(kap_rows).to_string())

# --- bootstrap CIs for the key representation-level contrasts ---
rng = np.random.default_rng(RNG_SEED)
B = 1000
n = len(y_te)
boot_rows = []
def _stats(yv, sv):
    e = EmpiricalOPR(yv, sv)
    return dict(auc=e.auc(), brier=e.brier(),
                nb05=e.net_benefit(0.05), nb10=e.net_benefit(0.10))
diffs = {("M1_logistic", "M2_distorted"): (p1, p2),
         ("M3_recalibrated", "M2_distorted"): (p3, p2),
         ("M4_gbm", "M1_logistic"): (p4, p1)}
for (a, b), (sa, sb) in diffs.items():
    ds = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y_te[idx]
        if yb.sum() in (0, n):
            continue
        da, db = _stats(yb, sa[idx]), _stats(yb, sb[idx])
        ds.append({k: da[k] - db[k] for k in da})
    ds = pd.DataFrame(ds)
    for k in ds.columns:
        boot_rows.append(dict(contrast=f"{a} - {b}", metric=k,
                              mean=float(ds[k].mean()),
                              lo=float(ds[k].quantile(0.025)),
                              hi=float(ds[k].quantile(0.975))))
pd.DataFrame(boot_rows).to_csv(
    os.path.join(ROOT, "analysis/realdata_bootstrap.csv"), index=False)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "analysis/realdata_metrics.csv"), index=False)
pd.DataFrame([dict(n_encounters=n_encounters, n_cohort=len(df),
                   n_events=int(y.sum()), prevalence_cohort=float(y.mean()),
                   n_train=len(y_tr), n_test=len(y_te),
                   prevalence=float(y_te.mean()))]).to_csv(
    os.path.join(ROOT, "analysis/realdata_cohort.csv"), index=False)
np.savez(os.path.join(ROOT, "data/processed/realdata_scores.npz"),
         y=y_te, **{f"s_{k}": v for k, v in models.items()})
print(out.to_string())
print("cohort n:", len(df), "events:", int(y.sum()),
      "prevalence:", round(y.mean(), 4))
