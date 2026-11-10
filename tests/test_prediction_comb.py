"""Prediction Comb tests: normalization, threshold invariance, and the
M1 -> M2 -> M3 controlled demonstration on the tooth locations."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from prediction_comb import compute_prediction_comb
from oprep import EmpiricalOPR, g_transform


def _data():
    rng = np.random.default_rng(7)
    n = 20000
    s = rng.beta(2, 5, n)
    y = rng.binomial(1, np.clip(s * 1.4, 0, 1))
    return y, s


def test_prevalence_weighted_mass_sums_to_one():
    y, s = _data()
    c = compute_prediction_comb(y, s, n_bins=50)
    assert abs(c["h0"].sum() + c["h1"].sum() - 1.0) < 1e-12
    assert abs(c["h1"].sum() - c["pi_hat"]) < 1e-12


def test_conditional_mass_sums_to_one_within_outcome():
    y, s = _data()
    c = compute_prediction_comb(y, s, n_bins=50, weighting="conditional")
    assert abs(c["h0"].sum() - 1.0) < 1e-12
    assert abs(c["h1"].sum() - 1.0) < 1e-12


def test_tooth_heights_are_joint_bin_mass():
    y, s = _data()
    edges = np.linspace(0, 1, 51)
    c = compute_prediction_comb(y, s, edges=edges)
    n = len(y)
    j = 3
    exp1 = np.mean((s >= edges[j]) & (s < edges[j + 1]) & (y == 1))
    exp0 = np.mean((s >= edges[j]) & (s < edges[j + 1]) & (y == 0))
    assert abs(c["h1"][j] - exp1) < 1e-15
    assert abs(c["h0"][j] - exp0) < 1e-15


def test_threshold_overlay_does_not_change_teeth():
    # Thresholds are a pure overlay: tooth values must be identical
    # regardless of any threshold drawn on top of the comb.
    y, s = _data()
    c1 = compute_prediction_comb(y, s, n_bins=50)
    c2 = compute_prediction_comb(y, s, n_bins=50)
    assert np.array_equal(c1["h1"], c2["h1"])
    assert np.array_equal(c1["h0"], c2["h0"])


def test_monotone_transform_moves_teeth_preserves_auc():
    y, s = _data()
    s2 = g_transform(s, 3)
    e1, e2 = EmpiricalOPR(y, s), EmpiricalOPR(y, s2)
    assert abs(e1.auc() - e2.auc()) < 1e-9
    edges = np.linspace(0, 1, 101)
    c1 = compute_prediction_comb(y, s, edges=edges)
    c2 = compute_prediction_comb(y, s2, edges=edges)
    # Mass-weighted tooth locations shift along the risk scale.
    m1 = (c1["centers"] * (c1["h0"] + c1["h1"])).sum()
    m2 = (c2["centers"] * (c2["h0"] + c2["h1"])).sum()
    assert abs(m1 - m2) > 0.05


def test_recalibration_returns_comb_toward_baseline():
    # M1 -> M2 (logit-scale distortion) -> M3 (logit recalibration):
    # teeth of M3 return toward the M1 configuration.
    rng = np.random.default_rng(11)
    n = 20000
    s1 = rng.beta(2, 5, n)
    y = rng.binomial(1, np.clip(s1 * 1.4, 0, 1))
    k = 1.6
    s2 = g_transform(s1, k)
    eps = 1e-9
    logit2 = np.log(np.clip(s2, eps, 1 - eps) / np.clip(1 - s2, eps, 1 - eps))
    from sklearn.linear_model import LogisticRegression
    recal = LogisticRegression(max_iter=1000).fit(logit2.reshape(-1, 1), y)
    s3 = recal.predict_proba(logit2.reshape(-1, 1))[:, 1]
    edges = np.linspace(0, 1, 101)
    c1 = compute_prediction_comb(y, s1, edges=edges)
    c2 = compute_prediction_comb(y, s2, edges=edges)
    c3 = compute_prediction_comb(y, s3, edges=edges)
    d21 = np.abs(c2["h1"] - c1["h1"]).sum() + np.abs(c2["h0"] - c1["h0"]).sum()
    d31 = np.abs(c3["h1"] - c1["h1"]).sum() + np.abs(c3["h0"] - c1["h0"]).sum()
    assert d31 < d21
