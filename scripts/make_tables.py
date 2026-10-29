"""Generate tables (CSV + docx-ready data structures)."""
import os, sys, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)

# ---- Table 1: metric recoverability --------------------------------
t1 = [
    ["ROC curve", "TPR(t), FPR(t) curve", "Yes — TPR(t)=1-F1(t), FPR(t)=1-F0(t)", "None", "No (rank quantities only)", "Prediction-scale location and spread (rank information only kept)", "Rank-invariant by construction"],
    ["AUC / c-statistic", "Scalar, P(S1>S0)", "Yes — integral of ROC curve", "None", "No", "All scale and class-conditional structure", "Prevalence-invariant"],
    ["Calibration curve", "m(s)=P(Y=1|S=s)", "Yes — pi f1/(pi f1+(1-pi) f0)", "None", "Yes (pi enters via Bayes weights)", "-", "Identity NW/loess estimators agree exactly"],
    ["Calibration-in-the-large", "Intercept at slope 1", "Yes — compare mean s to pi", "None", "Yes", "-", ""],
    ["Calibration slope/intercept", "Logistic refit on logit(s)", "Yes", "None", "Yes", "-", ""],
    ["ICI / E50 / E90", "Functionals of |m(s)-s| over f(s)", "Yes", "None", "Yes", "-", "Austin & Steyerberg metrics are functionals of R"],
    ["Brier score", "E[(Y-S)^2]", "Yes — integral over C(y,s)", "None", "Yes", "-", "Decomposable via Murphy partition"],
    ["Log loss", "E[-y log s - (1-y) log(1-s)]", "Yes — integral over C(y,s)", "None", "Yes", "-", "Proper scoring rule"],
    ["Sensitivity / specificity", "Rates at threshold t", "Yes — integrate C over regions", "Threshold t", "No (within-class rates)", "-", ""],
    ["PPV / NPV", "Predictive values at t", "Yes", "Threshold t", "Yes", "-", "Prevalence-dependent"],
    ["Confusion matrix", "2x2 counts at t", "Yes — N x joint masses", "Threshold t", "Yes (joint masses scale with pi)", "All off-threshold information", ""],
    ["Cohen's kappa", "Chance-corrected agreement at t", "Yes — from the 2x2 joint masses", "Threshold t (classification rule)", "Yes (marginals pi and q(t))", "All off-threshold information", "Agreement-optimal t is not necessarily clinically optimal"],
    ["Risk distributions", "f(s) per outcome", "Yes — is the representation", "None", "Marginal f(s) yes; f0, f1 no", "-", ""],
    ["Decision curve / net benefit", "NB(pt) curve", "Yes — TP(pt) - FP(pt) pt/(1-pt)", "External threshold preference / loss ratio", "Yes", "-", "Utility is not determined by data alone"],
]
pd.DataFrame(t1, columns=[
    "Evaluation domain", "Conventional representation/metric",
    "Recoverable from OPR?", "Additional information required",
    "Prevalence-dependent?",
    "Information discarded by conventional summary", "Notes"]
).to_csv(os.path.join(OUT, "table1_recoverability.csv"), index=False)

# ---- Table 2: simulation summary -----------------------------------
df = pd.read_csv(os.path.join(ROOT, "analysis/simulation_summary.csv"))
cols = ["scenario", "model", "prevalence", "auc", "brier", "logloss",
        "sens@0.1", "spec@0.1", "sens@0.25", "spec@0.25", "nb@0.1",
        "overlap_integral"]
df[cols].round(3).to_csv(os.path.join(OUT, "table2_simulation.csv"), index=False)

# ---- Table 3: real-data metrics ------------------------------------
rd = pd.read_csv(os.path.join(ROOT, "analysis/realdata_metrics.csv"))
cols3 = ["model", "n_test", "prevalence", "auc", "brier", "logloss",
         "calib_intercept", "calib_slope", "ICI",
         "sens@0.05", "spec@0.05", "nb@0.02", "nb@0.05", "nb@0.1"]
rd[cols3].round(4).to_csv(os.path.join(OUT, "table4_realdata.csv"), index=False)

# ---- Table 4: finite sample ----------------------------------------
fs = pd.read_csv(os.path.join(ROOT, "analysis/finite_sample.csv"))
g = fs.groupby("n").agg(
    auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
    brier_mean=("brier", "mean"), brier_sd=("brier", "std"),
    f0_l1_hist=("f0_l1_hist", "mean"), f0_l1_kde=("f0_l1_kde", "mean"),
    f1_l1_hist=("f1_l1_hist", "mean"), f1_l1_kde=("f1_l1_kde", "mean"))
g.round(4).to_csv(os.path.join(OUT, "table3_finite_sample.csv"))
print(g.round(4))
print("tables written")
