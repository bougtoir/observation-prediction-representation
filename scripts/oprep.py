"""oprep: Observation-Prediction Representation (OPR).

Primary representation: R = {f0(s), f1(s), pi}, the outcome-conditional
score densities of S in [0,1] together with prevalence pi = P(Y=1).
Equivalent to the joint C(y,s): C(0,s)=(1-pi) f0(s), C(1,s)=pi f1(s).

All standard evaluation objects are functionals of R:
  ROC / AUC, calibration curve m(s), threshold-specific confusion
  matrices, proper scoring-rule expectations (Brier, log loss), and
  decision-analytic net benefit given an external threshold preference.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq


# ----------------------------------------------------------------- #
# Representation objects
# ----------------------------------------------------------------- #

class DensityOPR:
    """Population-level OPR on a fixed grid of s in [0,1].

    Parameters
    ----------
    grid : (m,) array of s values
    f0, f1 : (m,) densities evaluated on grid (integrate to ~1)
    pi : prevalence P(Y=1)
    """

    def __init__(self, grid, f0, f1, pi):
        self.grid = np.asarray(grid, float)
        self.f0 = np.asarray(f0, float)
        self.f1 = np.asarray(f1, float)
        self.pi = float(pi)
        self.F0 = np.concatenate([[0], cumulative_trapezoid(self.f0, self.grid)])
        self.F1 = np.concatenate([[0], cumulative_trapezoid(self.f1, self.grid)])

    # -- derived functionals --------------------------------------

    def roc(self):
        """Return (fpr, tpr) curves on the grid."""
        fpr = 1.0 - self.F0
        tpr = 1.0 - self.F1
        order = np.argsort(fpr)
        return fpr[order], tpr[order]

    def auc(self):
        """AUC = P(S_1 > S_0) = integral F0(s) f1(s) ds."""
        fpr, tpr = self.roc()
        return trapezoid(tpr, fpr)

    def calib_m(self):
        """m(s) = P(Y=1|S=s) = pi f1 / (pi f1 + (1-pi) f0)."""
        num = self.pi * self.f1
        den = num + (1 - self.pi) * self.f0
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den > 0, num / den, np.nan)
        return m

    def confusion(self, t):
        """Threshold-t confusion matrix as joint probabilities + rates."""
        i = np.searchsorted(self.grid, t)
        F0_t, F1_t = self.F0[min(i, len(self.grid) - 1)], self.F1[min(i, len(self.grid) - 1)]
        tp = self.pi * (1 - F1_t)
        fn = self.pi * F1_t
        fp = (1 - self.pi) * (1 - F0_t)
        tn = (1 - self.pi) * F0_t
        sens = tp / (tp + fn) if tp + fn > 0 else np.nan
        spec = tn / (tn + fp) if tn + fp > 0 else np.nan
        ppv = tp / (tp + fp) if tp + fp > 0 else np.nan
        npv = tn / (tn + fn) if tn + fn > 0 else np.nan
        return dict(t=t, TP=tp, FN=fn, FP=fp, TN=tn,
                    sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv,
                    accuracy=tp + tn, balanced_accuracy=(sens + spec) / 2)

    def brier(self):
        """E[(Y-S)^2] = ∫ C(0,s) s^2 ds + ∫ C(1,s) (1-s)^2 ds."""
        c0 = (1 - self.pi) * self.f0
        c1 = self.pi * self.f1
        return trapezoid(c0 * self.grid**2, self.grid) + trapezoid(c1 * (1 - self.grid)**2, self.grid)

    def logloss(self, eps=1e-15):
        """E[-Y log S - (1-Y) log(1-S)]."""
        c0 = (1 - self.pi) * self.f0
        c1 = self.pi * self.f1
        s = np.clip(self.grid, eps, 1 - eps)
        return -(trapezoid(c1 * np.log(s), self.grid) + trapezoid(c0 * np.log(1 - s), self.grid))

    def net_benefit(self, pt):
        """NB(pt) = TP - FP * pt/(1-pt)  (per-patient scale)."""
        c = self.confusion(pt)
        odds = pt / (1 - pt)
        return c["TP"] - c["FP"] * odds

    def overlap(self):
        """O(s) = min{(1-pi) f0(s), pi f1(s)}; integral = Bayes-error rate."""
        o = np.minimum((1 - self.pi) * self.f0, self.pi * self.f1)
        return o, trapezoid(o, self.grid)


class EmpiricalOPR:
    """Finite-sample OPR estimated from (y, s) data.

    Estimators
    ----------
    'ecdf'      : cumulative representation F0,F1 (no smoothing; exact
                  finite-sample information). Densities via differencing
                  are not needed for ROC/confusion/NB/Brier.
    'histogram' : binned densities on `n_bins` equal bins.
    'kde'       : kernel density on the logit scale (boundary-corrected),
                  back-transformed to [0,1].
    """

    def __init__(self, y, s, method="ecdf", n_bins=20, bandwidth=None):
        self.y = np.asarray(y).astype(int)
        self.s = np.asarray(s, float)
        self.method = method
        self.n_bins = n_bins
        self.s1 = np.sort(self.s[self.y == 1])
        self.s0 = np.sort(self.s[self.y == 0])
        self.n1, self.n0 = len(self.s1), len(self.s0)
        self.n = self.n1 + self.n0
        self.pi = self.n1 / self.n
        self._bandwidth = bandwidth

    # -- ECDF ------------------------------------------------------

    def F(self, val, cls):
        """P(S <= val | Y=cls) via empirical CDF (mid-point convention)."""
        arr = self.s1 if cls == 1 else self.s0
        return (np.searchsorted(arr, val, side="right") +
                np.searchsorted(arr, val, side="left")) / (2 * len(arr))

    def F0(self, val): return self.F(val, 0)
    def F1(self, val): return self.F(val, 1)

    # -- density estimators ----------------------------------------

    def _kde(self, arr):
        """Gaussian KDE on logit scale with reflection-free transform."""
        from scipy.stats import gaussian_kde
        a = np.clip(arr, 1e-6, 1 - 1e-6)
        z = np.log(a / (1 - a))
        h = self._bandwidth
        kde = gaussian_kde(z, bw_method=h)
        return kde  # density in logit space

    def density(self, grid):
        """Return f0(grid), f1(grid) for the chosen method."""
        grid = np.asarray(grid, float)
        if self.method == "histogram":
            edges = np.linspace(0, 1, self.n_bins + 1)
            h0, _ = np.histogram(self.s0, bins=edges)
            h1, _ = np.histogram(self.s1, bins=edges)
            cen = (edges[:-1] + edges[1:]) / 2
            w = edges[1] - edges[0]
            f0 = np.interp(grid, cen, h0 / (self.n0 * w), left=0, right=0)
            f1 = np.interp(grid, cen, h1 / (self.n1 * w), left=0, right=0)
            return f0, f1
        if self.method == "kde":
            k0, k1 = self._kde(self.s0), self._kde(self.s1)
            g = np.clip(grid, 1e-6, 1 - 1e-6)
            z = np.log(g / (1 - g))
            # Jacobian: f_s(s) = f_z(logit s) / (s (1-s))
            f0 = k0(z) / (g * (1 - g))
            f1 = k1(z) / (g * (1 - g))
            f0[grid <= 0] = 0
            f1[grid <= 0] = 0
            f0[grid >= 1] = 0
            f1[grid >= 1] = 0
            return f0, f1
        raise ValueError("density available for 'histogram' and 'kde'")

    # -- derived functionals (direct from data) ---------------------

    def roc(self):
        """FPR/TPR curves from ECDF at the observed score values."""
        thr = np.r_[1.0 + 1e-9, np.unique(self.s)[::-1], -1e-9]
        tpr = 1 - np.searchsorted(self.s1, thr, "left") / self.n1
        fpr = 1 - np.searchsorted(self.s0, thr, "left") / self.n0
        return fpr, tpr

    def auc(self):
        fpr, tpr = self.roc()
        return trapezoid(tpr, fpr)

    def confusion(self, t):
        tp = np.mean((self.s >= t) & (self.y == 1))
        fn = np.mean((self.s < t) & (self.y == 1))
        fp = np.mean((self.s >= t) & (self.y == 0))
        tn = np.mean((self.s < t) & (self.y == 0))
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        return dict(t=t, TP=tp, FN=fn, FP=fp, TN=tn,
                    sensitivity=sens, specificity=spec,
                    PPV=tp / (tp + fp) if tp + fp else np.nan,
                    NPV=tn / (tn + fn) if tn + fn else np.nan,
                    accuracy=tp + tn,
                    balanced_accuracy=np.nanmean([sens, spec]))

    def brier(self):
        return np.mean((self.y - self.s) ** 2)

    def logloss(self, eps=1e-15):
        s = np.clip(self.s, eps, 1 - eps)
        return -np.mean(self.y * np.log(s) + (1 - self.y) * np.log(1 - s))

    def net_benefit(self, pt):
        c = self.confusion(pt)
        return c["TP"] - c["FP"] * pt / (1 - pt)

    def calib_m(self, grid, bandwidth=0.4):
        """m(s) = pi f1/(pi f1 + (1-pi) f0) with NW kernel estimates of
        the weighted class densities. Algebraically identical to the
        direct NW regression estimator local_logistic_calib()."""
        grid = np.asarray(grid, float)
        u = (grid[:, None] - self.s[None, :]) / bandwidth
        w = np.exp(-0.5 * u**2)
        # pi*f1(s) ~ (1/n) sum_{y=1} K ; (1-pi)*f0(s) ~ (1/n) sum_{y=0} K
        num = w[:, self.y == 1].sum(1) / self.n
        den = w.sum(1) / self.n
        return np.where(den > 0, num / den, np.nan)

    def local_logistic_calib(self, grid, bandwidth=0.4):
        """Direct nonparametric estimate of m(s) = P(Y=1|S=s) via
        Nadaraya-Watson on the s scale (Gaussian kernel)."""
        grid = np.asarray(grid, float)
        u = (grid[:, None] - self.s[None, :]) / bandwidth
        w = np.exp(-0.5 * u**2)
        num = w @ self.y
        den = w.sum(1)
        return np.where(den > 0, num / den, np.nan)


# ----------------------------------------------------------------- #
# Monotone-transformation utilities (ROC-invariance demonstrations)
# ----------------------------------------------------------------- #

def g_transform(s, k):
    """g(s) = s^k / (s^k + (1-s)^k): strictly increasing for k>0.
    Preserves ROC/AUC exactly, distorts calibration when k != 1."""
    s = np.clip(np.asarray(s, float), 0, 1)
    a = s**k
    return a / (a + (1 - s)**k)


def beta_density(grid, a, b):
    from scipy.stats import beta
    return beta.pdf(np.clip(grid, 1e-9, 1 - 1e-9), a, b)
