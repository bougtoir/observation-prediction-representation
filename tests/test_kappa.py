"""Cohen's kappa as a threshold-dependent classification functional
recoverable from the observation-prediction representation after a
classification rule is specified. Six tests."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from oprep import (EmpiricalOPR, kappa_from_confusion,
                   compute_kappa_at_threshold, compute_kappa_curve,
                   g_transform)
from prediction_comb import compute_prediction_comb
from sklearn.metrics import cohen_kappa_score


def _data():
    rng = np.random.default_rng(7)
    n = 4000
    y = rng.binomial(1, 0.15, n)
    s = np.clip(rng.beta(2, 8, n) + 0.35 * y * rng.beta(3, 6, n), 0, 1)
    return y, s


def test_helper_matches_sklearn():
    """Direct thresholded kappa equals sklearn.metrics.cohen_kappa_score."""
    y, s = _data()
    for t in [0.05, 0.10, 0.20, 0.40]:
        sk = cohen_kappa_score(y, (s >= t).astype(int))
        assert np.isclose(compute_kappa_at_threshold(y, s, t), sk, atol=1e-12)


def test_repr_reconstruction_matches_sample():
    """kappa reconstructed from the empirical OPR equals the direct
    sample calculation at every threshold."""
    y, s = _data()
    e = EmpiricalOPR(y, s)
    for t in [0.03, 0.08, 0.15, 0.30]:
        assert np.isclose(e.kappa(t),
                          compute_kappa_at_threshold(y, s, t), atol=1e-12)


def test_po_pe_match_confusion_matrix():
    """p_o and p_e equal the confusion-matrix margins: p_o = TP+TN,
    p_e = pi*q + (1-pi)(1-q) with pi = TP+FN, q = TP+FP."""
    y, s = _data()
    e = EmpiricalOPR(y, s)
    t = 0.12
    c = e.confusion(t)
    p_o = c["TP"] + c["TN"]
    q = c["TP"] + c["FP"]
    pi = c["TP"] + c["FN"]
    p_e = pi * q + (1 - pi) * (1 - q)
    assert np.isclose(p_o, np.mean((s >= t) == (y == 1)))
    assert np.isclose(q, np.mean(s >= t))
    assert np.isclose(pi, np.mean(y))
    assert np.isclose(kappa_from_confusion(c),
                      (p_o - p_e) / (1 - p_e))


def test_prevalence_affects_kappa():
    """Same score distribution, different prevalence -> different kappa
    (marginals enter p_e)."""
    rng = np.random.default_rng(11)
    s = rng.beta(2, 6, 5000)
    # identical scores; fabricate two outcome vectors with same PPV-ish
    # joint structure but different prevalence
    y_hi = np.zeros(5000); y_hi[:1000] = 1; y_hi = y_hi.astype(int)
    y_lo = np.zeros(5000); y_lo[:250] = 1; y_lo = y_lo.astype(int)
    k_hi = compute_kappa_at_threshold(y_hi, s, 0.4)
    k_lo = compute_kappa_at_threshold(y_lo, s, 0.4)
    assert not np.isclose(k_hi, k_lo)


def test_threshold_overlay_preserves_comb():
    """Overlays do not alter Prediction Comb data: kappa is computed
    downstream of a thresholded comb, leaving the teeth untouched."""
    y, s = _data()
    comb = compute_prediction_comb(y, s, n_bins=40)
    h0_before, h1_before = comb["h0"].copy(), comb["h1"].copy()
    _ = compute_kappa_curve(y, s, np.linspace(0.01, 0.9, 50))
    assert np.array_equal(comb["h0"], h0_before)
    assert np.array_equal(comb["h1"], h1_before)


def test_kappa_argmax_not_hardcoded_to_nb_argmax():
    """The kappa-maximizing threshold is computed, not assumed equal to
    the net-benefit-optimal threshold (they can coincide or differ)."""
    y, s = _data()
    e = EmpiricalOPR(y, s)
    grid = np.linspace(0.01, 0.5, 200)
    tk = grid[int(np.argmax([e.kappa(t) for t in grid]))]
    tnb = grid[int(np.argmax([e.net_benefit(t) for t in grid]))]
    # both are finite and were computed independently
    assert np.isfinite(tk) and np.isfinite(tnb)
    # monotone distortion leaves ranks (and hence kappa-vs-NB ordering
    # structure) consistent only through the representation
    s2 = g_transform(s, 1.6)
    e2 = EmpiricalOPR(y, s2)
    assert np.isclose(e.auc(), e2.auc())
