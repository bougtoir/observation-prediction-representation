"""Dedicated test for the corrected logit-scale M3 recalibration."""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from oprep import EmpiricalOPR, g_transform
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(42)
N = 20000
K = 1.6


@pytest.fixture(scope="module")
def synthetic():
    """Calibrated p1 from a known binary-outcome process."""
    logit_p1 = RNG.normal(-2.0, 1.2, N)
    p1 = 1 / (1 + np.exp(-logit_p1))
    y = RNG.binomial(1, p1)
    return y, p1


def test_logit_identity_of_g(synthetic):
    y, p1 = synthetic
    p2 = g_transform(p1, K)
    eps = 1e-12
    l2 = np.log(np.clip(p2, eps, 1 - eps) / np.clip(1 - p2, eps, 1 - eps))
    l1 = np.log(np.clip(p1, eps, 1 - eps) / np.clip(1 - p1, eps, 1 - eps))
    assert np.allclose(l2, K * l1, atol=1e-6)


def test_logit_recalibration_recovers_inverse_slope(synthetic):
    y, p1 = synthetic
    p2 = g_transform(p1, K)
    eps = 1e-9
    l2 = np.log(np.clip(p2, eps, 1 - eps) / np.clip(1 - p2, eps, 1 - eps))
    recal = LogisticRegression(max_iter=1000)
    recal.fit(l2.reshape(-1, 1), y)
    a, b = recal.intercept_[0], recal.coef_[0, 0]
    assert abs(b - 1.0 / K) < 0.08
    assert abs(a) < 0.15
    p3 = recal.predict_proba(l2.reshape(-1, 1))[:, 1]
    brier2 = EmpiricalOPR(y, p2).brier()
    brier3 = EmpiricalOPR(y, p3).brier()
    assert brier3 < brier2


def test_auc_invariant_through_chain(synthetic):
    y, p1 = synthetic
    p2 = g_transform(p1, K)
    eps = 1e-9
    l2 = np.log(np.clip(p2, eps, 1 - eps) / np.clip(1 - p2, eps, 1 - eps))
    recal = LogisticRegression(max_iter=1000).fit(l2.reshape(-1, 1), y)
    p3 = recal.predict_proba(l2.reshape(-1, 1))[:, 1]
    a1, a2, a3 = (EmpiricalOPR(y, s).auc() for s in (p1, p2, p3))
    assert abs(a1 - a2) < 1e-9
    assert abs(a1 - a3) < 1e-9
