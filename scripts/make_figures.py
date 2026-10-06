"""Publication figures (TIFF, 300+ dpi). Grayscale-friendly, patterned
fills, no decorative elements."""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from oprep import EmpiricalOPR, g_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.linewidth": 0.6, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 100,
})

GRID = np.linspace(0.002, 0.998, 600)


def kde_dens(y, s, grid=GRID):
    e = EmpiricalOPR(y, s, method="kde")
    return e.density(grid), e.pi


def opr_panel(ax, y, s, bw=0.06, show_m=True, thr=None, label=None):
    """Mirrored OPR: f1 above axis, f0 below; m(s) overlay dotted."""
    (f0, f1), pi = kde_dens(y, s)
    ax.fill_between(GRID, 0, f1, color="0.25", alpha=0.9, label="Y=1")
    ax.fill_between(GRID, 0, -f0, color="0.85", hatch="///",
                    edgecolor="0.45", lw=0.3, label="Y=0")
    ax.axhline(0, color="black", lw=0.6)
    if show_m:
        e = EmpiricalOPR(y, s)
        m = e.calib_m(GRID, bandwidth=bw)
        ax.plot(GRID, m * f1.max() * 0.9, color="black", lw=1.2, ls=":",
                label="m(s) (scaled)")
    if thr is not None:
        ax.axvline(thr, color="black", lw=0.8, ls="--")
        ax.text(thr, ax.get_ylim()[1], " t", fontsize=7.5, va="top")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prediction score s")
    ax.set_ylabel("Density")
    if label:
        ax.set_title(label, fontsize=9)
    ax.legend(frameon=False, loc="upper right")


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".tiff"), dpi=400,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150)
    plt.close(fig)
    print("wrote", name)


P = lambda n: np.load(os.path.join(ROOT, "data/processed", f"scenario_{n}.npz"))

# ---- Figure 1: conceptual hierarchy (schematic) -------------------
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.axis("off")
boxes = [
    (0.5, 0.94, "Observed outcomes y and prediction scores s"),
    (0.5, 0.74, "Observation–prediction representation\n$R=\\{f_0(s),\\,f_1(s),\\,\\pi\\} \\;\\equiv\\; C(y,s)$"),
    (0.5, 0.44, "Statistical performance functionals\n(discrimination · calibration · classification · overall scores)"),
    (0.5, 0.14, "Decision-analytic utility\n(representation + external preferences $\\rightarrow$ net benefit)"),
]
for x, y0, t in boxes:
    ax.text(x, y0, t, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="0.96", ec="black", lw=0.7))
for ya, yb in [(0.885, 0.80), (0.685, 0.50), (0.385, 0.20)]:
    ax.annotate("", xy=(0.5, yb), xytext=(0.5, ya),
                arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate("", xy=(0.80, 0.14), xytext=(0.80, 0.50),
            arrowprops=dict(arrowstyle="-", lw=0.9, ls="--"))
ax.text(0.83, 0.32, "decision preferences,\nthreshold, loss ratio", fontsize=7.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
savefig(fig, "FIGURE_1")

# ---- Figure 2: OPR display, archetypes (A perfect, B random,
#      C calibrated vs g-transformed) -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
d = P("A_perfect"); opr_panel(axs[0, 0], d["y_model"], d["s_model"], label="A. Near-perfect prediction")
d = P("B_random"); opr_panel(axs[0, 1], d["y_model"], d["s_model"], label="B. Non-informative prediction")
d = P("C_same_roc_diff_calib")
opr_panel(axs[1, 0], d["y_calibrated"], d["s_calibrated"], label="C1. Calibrated model")
opr_panel(axs[1, 1], d["y_g_transformed_k3"], d["s_g_transformed_k3"],
          label="C2. Same ROC, distorted scale")
fig.tight_layout()
savefig(fig, "FIGURE_3")

# ---- Figure 3: C — identical ROC, different m(s) ------------------
d = P("C_same_roc_diff_calib")
fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9))
e_cal = EmpiricalOPR(d["y_calibrated"], d["s_calibrated"])
e_tr = EmpiricalOPR(d["y_g_transformed_k3"], d["s_g_transformed_k3"])
fpr, tpr = e_cal.roc(); axs[0].plot(fpr, tpr, "k-", lw=1.0, label="calibrated")
fpr, tpr = e_tr.roc(); axs[0].plot(fpr, tpr, "k--", lw=1.0, label="g(s), k=3")
axs[0].plot([0, 1], [0, 1], color="0.7", lw=0.6)
axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].legend(frameon=False)
axs[0].set_title("ROC: identical")
axs[1].plot(GRID, e_cal.calib_m(GRID, bw_smooth := 0.08), "k-", lw=1.0, label="calibrated")
axs[1].plot(GRID, e_tr.calib_m(GRID, 0.08), "k--", lw=1.0, label="g(s)")
axs[1].plot([0, 1], [0, 1], color="0.7", lw=0.6)
axs[1].set_xlabel("s"); axs[1].set_ylabel("m(s)")
axs[1].set_title("Calibration curve"); axs[1].legend(frameon=False)
# densities
(f0a, f1a), _ = kde_dens(d["y_calibrated"], d["s_calibrated"])
(f0b, f1b), _ = kde_dens(d["y_g_transformed_k3"], d["s_g_transformed_k3"])
axs[2].plot(GRID, f1a, "k-", lw=0.9, label="f1 calib")
axs[2].plot(GRID, f1b, "k--", lw=0.9, label="f1 g(s)")
axs[2].plot(GRID, f0a, "k-", lw=0.9, alpha=0.45, label="f0 calib")
axs[2].plot(GRID, f0b, "k--", lw=0.9, alpha=0.45, label="f0 g(s)")
axs[2].set_xlabel("s"); axs[2].set_ylabel("Density")
axs[2].set_title("Score distributions"); axs[2].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
savefig(fig, "FIGURE_2")

# ---- Figure 4: D — same AUC, different overlap location -----------
d = P("D_overlap_location")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
opr_panel(axs[0], d["y_mid_overlap"], d["s_mid_overlap"], show_m=False, label="Mid-scale overlap")
opr_panel(axs[1], d["y_tail_overlap"], d["s_tail_overlap"], show_m=False, label="Tail-located overlap")
fig.tight_layout()
savefig(fig, "FIGURE_4")

# ---- Figure 5: F + G — prevalence & transport ---------------------
fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9))
# F: joint C(y,s) vs conditional view under prevalence shift
ax = axs[0]
for prev, ls in [("0.05", "-"), ("0.2", "--"), ("0.5", ":")]:
    d = P(f"F_prevalence_{prev}")
    y, s = d["y_model"], d["s_model"]
    (f0, f1), _ = kde_dens(y, s)
    e = EmpiricalOPR(y, s)
    ax.plot(GRID, e.pi * f1, ls, color="black", lw=1.0,
            label=f"pi={prev}")
ax.set_xlabel("s"); ax.set_ylabel("C(1,s) = pi f1(s)")
ax.set_title("F. Joint mass shifts with prevalence"); ax.legend(frameon=False)
# G: calibration curve source vs shifted
d = P("G_transport")
ax = axs[1]
for key, lab, ls in [("source_domain", "source", "-"), ("target_shifted", "transported", "--")]:
    e = EmpiricalOPR(d[f"y_{key}"], d[f"s_{key}"])
    ax.plot(GRID, e.calib_m(GRID, 0.08), ls, color="black", lw=1.0, label=lab)
ax.plot([0, 1], [0, 1], color="0.7", lw=0.6)
ax.set_xlabel("s"); ax.set_ylabel("m(s)"); ax.set_title("G. Calibration after shift")
ax.legend(frameon=False)
# H: NB difference in a threshold band
d = P("H_threshold_band")
ax = axs[2]
pts = np.linspace(0.02, 0.6, 120)
for key, ls, lab in [("model1", "-", "model 1"), ("model2", "--", "model 2")]:
    e = EmpiricalOPR(d[f"y_{key}"], d[f"s_{key}"])
    ax.plot(pts, [e.net_benefit(p) for p in pts], ls, color="black", lw=1.0, label=lab)
ax.axvspan(0.30, 0.45, color="0.9")
ax.set_xlabel("Threshold probability pt"); ax.set_ylabel("Net benefit")
ax.set_title("H. Net benefit differs locally"); ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "FIGURE_5")

# ---- Figure 6: real application ------------------------------------
d = np.load(os.path.join(ROOT, "data/processed/realdata_scores.npz"))
y = d["y"]
fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.4))
for j, (key, lab) in enumerate([("s_M1_logistic", "M1 logistic"),
                                ("s_M2_distorted", "M2 distorted"),
                                ("s_M3_recalibrated", "M3 recalibrated")]):
    s = d[key]
    e = EmpiricalOPR(y, s)
    opr_panel(axs[0, j], y, s, bw=0.05, label=lab)
    xmax = float(np.quantile(s, 0.995)) * 1.15
    axs[0, j].set_xlim(0, xmax)
    axs[0, j].set_ylim(axs[0, j].get_ylim()[0], axs[0, j].get_ylim()[1] * 1.1)
    # calibration curve over supported range only
    (f0, f1), _pi = kde_dens(y, s)
    fs = _pi * f1 + (1 - _pi) * f0
    m = e.calib_m(GRID, max(0.005, 0.15 * float(np.quantile(s, 0.995))))
    m = np.where(fs > fs.max() * 0.02, m, np.nan)
    axs[1, j].plot(GRID, m, "k-", lw=1.0)
    axs[1, j].plot([0, xmax], [0, xmax], color="0.7", lw=0.6)
    axs[1, j].set_xlim(0, xmax); axs[1, j].set_ylim(0, xmax)
    axs[1, j].set_xlabel("s"); axs[1, j].set_ylabel("m(s)")
fig.tight_layout()
savefig(fig, "FIGURE_7")

# decision curves for real data
fig, ax = plt.subplots(figsize=(4.6, 3.2))
pts = np.linspace(0.01, 0.30, 200)
for key, ls, lab in [("s_M1_logistic", "-", "M1 logistic"),
                     ("s_M2_distorted", "--", "M2 distorted"),
                     ("s_M3_recalibrated", ":", "M3 recalibrated")]:
    e = EmpiricalOPR(y, d[key])
    ax.plot(pts, [e.net_benefit(p) for p in pts], ls, color="black", lw=1.1, label=lab)
prev = y.mean()
ax.plot(pts, prev - (1 - prev) * pts / (1 - pts), color="0.6", lw=0.8,
        label="treat all")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("Threshold probability pt"); ax.set_ylabel("Net benefit")
ax.set_ylim(-0.02, 0.10); ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "FIGURE_8")

# ---- Figure 8: finite-sample estimator behaviour ------------------
fs = __import__("pandas").read_csv(os.path.join(ROOT, "analysis/finite_sample.csv"))
fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9))
g = fs.groupby("n")
axs[0].errorbar(g["auc"].mean().index, g["auc"].mean(), yerr=g["auc"].std(),
                fmt="k-o", ms=3, lw=0.8, capsize=2)
axs[0].axhline(0.917, color="0.6", ls="--", lw=0.8)
axs[0].set_xscale("log"); axs[0].set_xlabel("n"); axs[0].set_ylabel("AUC")
axs[0].set_title("AUC stability (ECDF)")
axs[1].errorbar(g["f0_l1_hist"].mean().index, g["f0_l1_hist"].mean(),
                yerr=g["f0_l1_hist"].std(), fmt="k-s", ms=3, lw=0.8, capsize=2, label="histogram")
axs[1].errorbar(g["f0_l1_kde"].mean().index, g["f0_l1_kde"].mean(),
                yerr=g["f0_l1_kde"].std(), fmt="k-o", ms=3, lw=0.8, capsize=2, label="KDE (logit)")
axs[1].set_xscale("log"); axs[1].set_xlabel("n")
axs[1].set_ylabel("L1 error in f0"); axs[1].set_title("Density estimator error")
axs[1].legend(frameon=False)
axs[2].errorbar(g["brier"].mean().index, g["brier"].mean(), yerr=g["brier"].std(),
                fmt="k-o", ms=3, lw=0.8, capsize=2)
axs[2].axhline(0.1372, color="0.6", ls="--", lw=0.8)
axs[2].set_xscale("log"); axs[2].set_xlabel("n"); axs[2].set_ylabel("Brier")
axs[2].set_title("Brier stability")
fig.tight_layout()
savefig(fig, "FIGURE_6")

print("done")
