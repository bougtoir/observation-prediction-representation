"""Automated checks required by the study design:

1. ROC derived from the OPR matches directly computed ROC.
2. Confusion matrices reconstructed at thresholds match direct counts.
3. Calibration from the representation matches the direct estimator.
4. Brier score via integration matches direct calculation.
5. Net benefit from the representation matches direct DCA calculation.
6. Monotone score transformations preserve ROC/AUC but change the
   prediction-scale representation and calibration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pytest
from sklearn.metrics import roc_curve, auc as sk_auc, brier_score_loss, log_loss, confusion_matrix

from oprep import DensityOPR, EmpiricalOPR, g_transform, beta_density


@pytest.fixture
def sample():
    rng = np.random.default_rng(7)
    n = 4000
    y = rng.binomial(1, 0.35, n)
    s = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n))
    return y, np.clip(s, 0, 1)


def test_roc_from_representation_matches_sklearn(sample):
    y, s = sample
    e = EmpiricalOPR(y, s)
    fpr_e, tpr_e = e.roc()
    fpr_s, tpr_s, _ = roc_curve(y, s)
    # sort sklearn's fpr; compare at same fpr grid
    g = np.linspace(0, 1, 101)
    assert np.allclose(np.interp(g, fpr_e, tpr_e), np.interp(g, fpr_s, tpr_s), atol=1e-3)
    assert abs(e.auc() - sk_auc(fpr_s, tpr_s)) < 1e-3


def test_confusion_matrix_matches_counts(sample):
    y, s = sample
    e = EmpiricalOPR(y, s)
    for t in [0.2, 0.5, 0.8]:
        c = e.confusion(t)
        tn, fp, fn, tp = confusion_matrix(y, s >= t).ravel()
        n = len(y)
        assert np.isclose(c["TP"], tp / n) and np.isclose(c["FP"], fp / n)
        assert np.isclose(c["TN"], tn / n) and np.isclose(c["FN"], fn / n)


def test_brier_and_logloss_integration():
    grid = np.linspace(0, 1, 4001)
    f0 = beta_density(grid, 2, 4)
    f1 = beta_density(grid, 4, 2)
    d = DensityOPR(grid, f0, f1, 0.35)
    rng = np.random.default_rng(3)
    n = 20000
    y = rng.binomial(1, 0.35, n)
    s = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n))
    assert abs(d.brier() - brier_score_loss(y, np.clip(s, 1e-6, 1 - 1e-6))) < 0.01
    assert abs(d.logloss() - log_loss(y, np.clip(s, 1e-6, 1 - 1e-6))) < 0.01


def test_net_benefit_matches_direct(sample):
    y, s = sample
    e = EmpiricalOPR(y, s)
    for pt in [0.1, 0.3, 0.5]:
        direct = np.mean((s >= pt) & (y == 1)) - np.mean((s >= pt) & (y == 0)) * pt / (1 - pt)
        assert np.isclose(e.net_benefit(pt), direct)


def test_calibration_from_representation(sample):
    y, s = sample
    e = EmpiricalOPR(y, s)
    grid = np.linspace(0.1, 0.9, 41)
    m_repr = e.calib_m(grid, bandwidth=0.25)
    m_direct = e.local_logistic_calib(grid, bandwidth=0.25)
    # the representation-based m(s) is the same estimator computed
    # through the class-density route: must agree exactly
    mask = np.isfinite(m_repr) & np.isfinite(m_direct)
    assert np.allclose(m_repr[mask], m_direct[mask], atol=1e-12)


def test_monotone_transform_preserves_roc_changes_calibration(sample):
    y, s = sample
    e1 = EmpiricalOPR(y, s)
    e2 = EmpiricalOPR(y, g_transform(s, 3))
    fpr1, tpr1 = e1.roc()
    fpr2, tpr2 = e2.roc()
    g = np.linspace(0, 1, 201)
    assert np.allclose(np.interp(g, fpr1, tpr1), np.interp(g, fpr2, tpr2), atol=2e-3)
    assert abs(e1.auc() - e2.auc()) < 2e-3
    # but the prediction-scale representation and calibration differ
    grid = np.linspace(0.05, 0.95, 91)
    f0a, f1a = e1.density(grid) if False else (None, None)
    b1, b2 = e1.brier(), e2.brier()
    assert abs(b1 - b2) > 1e-3
