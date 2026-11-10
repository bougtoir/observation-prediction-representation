"""Simulation study: scenarios A-J.

Each scenario produces (y, s) samples for one or more competing models.
For every scenario we compute the full evaluation battery:
  ROC/AUC, calibration curve m(s), Brier, log loss, confusion matrix at
  selected thresholds, net benefit at selected threshold probabilities,
  outcome-specific score densities, and the proposed OPR.

Outputs:
  analysis/simulation_summary.csv     one row per (scenario, model)
  data/processed/scenario_<name>.npz  raw (y, s) per model for figures
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from oprep import EmpiricalOPR, g_transform
from scipy.stats import beta as _beta

def beta_pdf(g, a, b): return _beta.pdf(g, a, b)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20261006)
N = 20000
THRESHOLDS = [0.10, 0.25, 0.50]
PT_GRID = [0.05, 0.10, 0.20, 0.30]


def metrics_row(scenario, model, y, s, n=None):
    e = EmpiricalOPR(y, s)
    o_lo = e  # ECDF object reused
    row = dict(scenario=scenario, model=model, n=len(y),
               prevalence=float(np.mean(y)),
               auc=e.auc(), brier=e.brier(), logloss=e.logloss())
    for t in THRESHOLDS:
        c = e.confusion(t)
        row[f"sens@{t}"] = c["sensitivity"]
        row[f"spec@{t}"] = c["specificity"]
        row[f"ppv@{t}"] = c["PPV"]
        row[f"acc@{t}"] = c["accuracy"]
    for pt in PT_GRID:
        row[f"nb@{pt}"] = e.net_benefit(pt)
    # overlap integral via histogram estimate
    grid = np.linspace(0.001, 0.999, 501)
    f0, f1 = EmpiricalOPR(y, s, method="kde").density(grid)
    o = np.minimum((1 - e.pi) * f0, e.pi * f1)
    row["overlap_integral"] = float(np.trapezoid(o, grid))
    return row


def save_scenario(name, **model_data):
    np.savez(os.path.join(ROOT, "data/processed", f"scenario_{name}.npz"),
             **{f"y_{k}": v[0] for k, v in model_data.items()},
             **{f"s_{k}": v[1] for k, v in model_data.items()})


def gen_calibrated(n, prev=0.3, sep=1.4, rng=RNG):
    """Logistic model: y|x ~ Bern(sigmoid(b0 + b1 x)), s = true p."""
    x = rng.normal(size=n)
    b0 = np.log(prev / (1 - prev)) - (sep**2) / 2 * 0  # center
    p = 1 / (1 + np.exp(-(b0 + sep * x)))
    y = rng.binomial(1, p)
    return y, p


rows = []

# ---- A: perfect prediction ---------------------------------------
y = RNG.binomial(1, 0.3, N)
s = np.where(y == 1, RNG.beta(60, 1.2, N), RNG.beta(1.2, 60, N))
rows.append(metrics_row("A_perfect", "model", y, s))
save_scenario("A_perfect", model=(y, s))

# ---- B: random / noninformative ----------------------------------
y = RNG.binomial(1, 0.3, N)
s = RNG.uniform(0, 1, N)
rows.append(metrics_row("B_random", "model", y, s))
save_scenario("B_random", model=(y, s))

# ---- C: same ROC & AUC, different calibration ---------------------
y, p = gen_calibrated(N)
s_mis = g_transform(p, 3.0)          # strictly monotone -> same ROC
rows.append(metrics_row("C_same_roc_diff_calib", "calibrated", y, p))
rows.append(metrics_row("C_same_roc_diff_calib", "g_transformed_k3", y, s_mis))
save_scenario("C_same_roc_diff_calib", calibrated=(y, p),
              g_transformed_k3=(y, s_mis))

# ---- D: similar AUC, overlap located differently ------------------
# Model D1: unimodal densities, overlap concentrated mid-scale.
#   f0 = Beta(3,4), f1 = Beta(4,3)  -> overlap near 0.5
# Model D2: U-shaped densities pushing overlap to the tails:
#   f0 = mixture Beta(1.2,5) [80%] + Beta(8,2) [20%]
#   f1 = mixture Beta(2,8) [20%] + Beta(5,1.2) [80%]
n1 = int(N * 0.3); n0 = N - n1
s0 = RNG.beta(3, 4, n0); s1 = RNG.beta(4, 3, n1)
s_d1 = np.r_[s0, s1]; y_d1 = np.r_[np.zeros(n0), np.ones(n1)]

# D2 'tail_overlap': Y=0 mass is 80% Beta(2.5,5) (low) + 20% Beta(5,2)
# (high); Y=1 is 20% Beta(2,5) (low) + 80% Beta(5,2.5) (high).
# Ambiguity sits in the tails; AUC tuned to match mid_overlap (~0.72).
m0 = RNG.random(n0) < 0.80
s0b = np.where(m0, RNG.beta(2.5, 5, n0), RNG.beta(5, 2, n0))
m1 = RNG.random(n1) < 0.20
s1b = np.where(m1, RNG.beta(2, 5, n1), RNG.beta(5, 2.5, n1))
s_d2 = np.r_[s0b, s1b]; y_d2 = y_d1.copy()

e1, e2 = EmpiricalOPR(y_d1, s_d1), EmpiricalOPR(y_d2, s_d2)
rows.append(metrics_row("D_overlap_location", "mid_overlap", y_d1, s_d1))
rows.append(metrics_row("D_overlap_location", "tail_overlap", y_d2, s_d2))
save_scenario("D_overlap_location", mid_overlap=(y_d1, s_d1),
              tail_overlap=(y_d2, s_d2))

# ---- E: different discrimination, similar calibration -------------
ya, pa = gen_calibrated(N, sep=1.6)
yb, pb = gen_calibrated(N, sep=0.6)
rows.append(metrics_row("E_disc_vs_calib", "strong", ya, pa))
rows.append(metrics_row("E_disc_vs_calib", "weak", yb, pb))
save_scenario("E_disc_vs_calib", strong=(ya, pa), weak=(yb, pb))

# ---- F: prevalence shift ------------------------------------------
# Same conditional structure, varying pi. Simulate from fixed f0,f1.
for prev in [0.05, 0.20, 0.50]:
    n1 = int(N * prev); n0 = N - n1
    s = np.r_[RNG.beta(2.5, 5, n0), RNG.beta(5, 2.5, n1)]
    y = np.r_[np.zeros(n0), np.ones(n1)]
    rows.append(metrics_row("F_prevalence", f"pi={prev}", y, s))
    save_scenario(f"F_prevalence_{prev}", model=(y, s))

# ---- G: transport / miscalibration under shift --------------------
# Source: calibrated p. Target: same ranking but miscalibrated absolute
# probabilities (risk inflation): p_target = sigmoid(logit(p)+delta)
def sigmoid(z): return 1 / (1 + np.exp(-z))
def logit(p): return np.log(p / (1 - p))
y_src, p_src = gen_calibrated(N)
x_t = RNG.normal(0.4, 1.1, N)                       # shifted covariates
p_t_true = sigmoid(np.log(0.3 / 0.7) + 1.4 * x_t)   # true target risk
y_t = RNG.binomial(1, p_t_true)
# model retains source ordering on x but absolute scale drifts low
p_t_mod = sigmoid(np.log(0.3 / 0.7) + 1.4 * (x_t - 0.4))
rows.append(metrics_row("G_transport", "source_domain", y_src, p_src))
rows.append(metrics_row("G_transport", "target_shifted", y_t, p_t_mod))
save_scenario("G_transport", source_domain=(y_src, p_src),
              target_shifted=(y_t, p_t_mod))

# ---- H: narrow clinically important threshold interval ------------
# Two models with similar AUC; M2 much better in t in [0.3, 0.45] only.
n1 = int(N * 0.3); n0 = N - n1
s0 = RNG.beta(2.5, 5, n0); s1 = RNG.beta(5, 2.5, n1)
s_m1 = np.r_[s0, s1]; y_h = np.r_[np.zeros(n0), np.ones(n1)]
# M2: separate the densities specifically in [0.3,0.45] band
s0b = s0.copy(); s1b = s1.copy()
band0 = (s0b > 0.30) & (s0b < 0.45)
band1 = (s1b > 0.30) & (s1b < 0.45)
s0b[band0] = RNG.beta(1.5, 6, band0.sum())      # push Y=0 scores lower
s1b[band1] = np.maximum(s1b[band1], RNG.beta(6, 1.5, band1.sum()))
s_m2 = np.r_[s0b, s1b]
rows.append(metrics_row("H_threshold_band", "model1", y_h, s_m1))
rows.append(metrics_row("H_threshold_band", "model2", y_h, s_m2))
save_scenario("H_threshold_band", model1=(y_h, s_m1), model2=(y_h, s_m2))

# ---- I: class imbalance -------------------------------------------
prev = 0.02
n1 = int(N * prev); n0 = N - n1
s = np.r_[RNG.beta(2, 5, n0), RNG.beta(5, 2, n1)]
y = np.r_[np.zeros(n0), np.ones(n1)]
rows.append(metrics_row("I_imbalance", "model", y, s))
save_scenario("I_imbalance", model=(y, s))

# ---- J: finite-sample estimator behavior --------------------------
# Repeated draws of a fixed population; compare ECDF / histogram / KDE
# AUC estimates and calibration-curve variability.
fs_rows = []
for n in [100, 200, 500, 1000, 5000, 20000]:
    for rep in range(50):
        n1 = int(n * 0.3); n0 = n - n1
        s = np.r_[RNG.beta(2.5, 5, n0), RNG.beta(5, 2.5, n1)]
        y = np.r_[np.zeros(n0), np.ones(n1)]
        e = EmpiricalOPR(y, s)
        grid = np.linspace(0.02, 0.98, 49)
        f0h, f1h = EmpiricalOPR(y, s, method="histogram", n_bins=15).density(grid)
        f0k, f1k = EmpiricalOPR(y, s, method="kde").density(grid)
        fs_rows.append(dict(n=n, rep=rep, auc=e.auc(), brier=e.brier(),
                            f0_l1_hist=float(np.trapezoid(np.abs(f0h - beta_pdf(grid, 2.5, 5)), grid)),
                            f1_l1_hist=float(np.trapezoid(np.abs(f1h - beta_pdf(grid, 5, 2.5)), grid)),
                            f0_l1_kde=float(np.trapezoid(np.abs(f0k - beta_pdf(grid, 2.5, 5)), grid)),
                            f1_l1_kde=float(np.trapezoid(np.abs(f1k - beta_pdf(grid, 5, 2.5)), grid))))
pd.DataFrame(fs_rows).to_csv(
    os.path.join(ROOT, "analysis/finite_sample.csv"), index=False)

pd.DataFrame(rows).to_csv(
    os.path.join(ROOT, "analysis/simulation_summary.csv"), index=False)
print("summary rows:", len(rows))
