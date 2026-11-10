"""Prediction Comb: binned graphical realization of the OPR.

The comb displays outcome-specific probability mass as teeth along the
original prediction scale s in [0,1]. For equal-width bins B_j:

  prevalence-weighted (primary display):
      upper tooth h1_j = P(S in B_j, Y=1) = pi * integral_{B_j} f1
      lower tooth h0_j = P(S in B_j, Y=0) = (1-pi) * integral_{B_j} f0
      sum_j (h1_j + h0_j) = 1

  class-conditional (option):
      h1_j = P(S in B_j | Y=1),  h0_j = P(S in B_j | Y=0)
      each sums to 1 within its outcome.

Tooth *heights* are probability mass per bin (not density); the x-axis
is the original predicted-risk scale and is never transformed.
"""

from __future__ import annotations

import numpy as np


def compute_prediction_comb(y, s, n_bins=50, edges=None,
                            weighting="prevalence"):
    """Return comb teeth for (y, s).

    Parameters
    ----------
    y, s : arrays of binary outcome and prediction score in [0,1]
    n_bins : number of equal-width bins on [0,1] (ignored if edges given)
    edges : explicit bin edges on the score scale
    weighting : 'prevalence' (joint mass) or 'conditional'

    Returns dict with edges, centers, h0, h1, pi_hat, weighting, n_bins.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.asarray(edges, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum()); n = len(y)
    c1, _ = np.histogram(s[y == 1], bins=edges)
    c0, _ = np.histogram(s[y == 0], bins=edges)
    pi_hat = n1 / n if n else np.nan
    if weighting == "prevalence":
        h1 = c1 / n
        h0 = c0 / n
    elif weighting == "conditional":
        h1 = c1 / n1 if n1 else np.zeros(len(edges) - 1)
        h0 = c0 / n0 if n0 else np.zeros(len(edges) - 1)
    else:
        raise ValueError("weighting must be 'prevalence' or 'conditional'")
    return dict(edges=edges, centers=(edges[:-1] + edges[1:]) / 2,
                h0=h0, h1=h1, pi_hat=pi_hat, weighting=weighting,
                n_bins=len(edges) - 1)


def plot_prediction_comb(ax, y, s, n_bins=50, edges=None,
                         weighting="prevalence", thresholds=None,
                         label=None, tooth_width_frac=0.6,
                         ymax=None):
    """Draw a Prediction Comb on `ax`.

    Upward teeth (dark solid) = Y=1 mass; downward teeth (light hatched)
    = Y=0 mass. Vertical dashed lines mark decision thresholds; they are
    overlaid only and never alter tooth values.
    """
    c = compute_prediction_comb(y, s, n_bins=n_bins, edges=edges,
                                weighting=weighting)
    bw = (c["edges"][1] - c["edges"][0]) * tooth_width_frac
    for x, h in zip(c["centers"], c["h1"]):
        ax.fill_between([x - bw / 2, x + bw / 2], 0, h,
                        color="0.25", lw=0)
    for x, h in zip(c["centers"], c["h0"]):
        ax.fill_between([x - bw / 2, x + bw / 2], 0, -h,
                        color="0.85", hatch="///", edgecolor="0.45", lw=0.3)
    ax.axhline(0, color="black", lw=0.6)
    if thresholds:
        for t in thresholds:
            ax.axvline(t, color="black", lw=0.8, ls="--", ymax=0.97)
    ax.set_xlim(c["edges"][0], c["edges"][-1])
    if ymax is not None:
        ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Prediction score s")
    ylab = "Joint mass P(S in bin, Y)" if weighting == "prevalence" \
        else "P(S in bin | Y)"
    ax.set_ylabel(ylab)
    if label:
        ax.set_title(label, fontsize=9)
    return c
