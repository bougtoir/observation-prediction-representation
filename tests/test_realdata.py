"""Tests for the real-data model comparison (diabetes readmission)."""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "data/processed/realdata_scores.npz")
RAW = os.path.join(ROOT, "data/raw/diabetic_data.csv")


def _pipeline_runnable():
    if os.path.exists(NPZ):
        return True
    if os.path.exists(RAW):
        import subprocess
        subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "scripts", "real_data_analysis.py")],
            check=True, cwd=ROOT)
        return True
    return False


if not os.path.exists(NPZ) and not _pipeline_runnable():
    if os.environ.get("OPR_ALLOW_DATA_SKIP") == "1":
        pytest.skip("real-data pipeline outputs unavailable "
                    "(OPR_ALLOW_DATA_SKIP=1)", allow_module_level=True)
    pytest.fail(
        "data/processed/realdata_scores.npz missing and "
        "data/raw/diabetic_data.csv not found. Run "
        "scripts/acquire_data.sh then `make all`, or set "
        "OPR_ALLOW_DATA_SKIP=1 to run CI without the clinical dataset.",
        pytrace=False)


class TestRealDataComparison:
    def setup_method(self):
        import sys
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from oprep import EmpiricalOPR
        d = np.load(NPZ)
        self.y = d["y"]
        self.s = {k[2:]: d[k] for k in d.files if k.startswith("s_")}
        self.E = EmpiricalOPR

    def test_rank_equivalence_m1_m2(self):
        assert self.E(self.y, self.s["M1_logistic"]).auc() == \
            self.E(self.y, self.s["M2_distorted"]).auc()

    def test_recovered_order_and_bands(self):
        aucs = {k: self.E(self.y, s).auc() for k, s in self.s.items()}
        assert aucs["M4_gbm"] > aucs["M1_logistic"]
        assert 0.55 < aucs["M4_gbm"] < 0.75

    def test_m3_recalibration_improves_ici(self):
        grid = np.linspace(0.005, 0.995, 199)
        icis = {}
        for k in ["M2_distorted", "M3_recalibrated", "M1_logistic"]:
            s = self.s[k]
            e = self.E(self.y, s)
            f0, f1 = self.E(self.y, s, method="kde").density(grid)
            fs = e.pi * f1 + (1 - e.pi) * f0
            sup = np.quantile(s, 0.995)
            m = e.calib_m(grid, bandwidth=max(0.005, 0.15 * sup))
            icis[k] = np.nansum(np.abs(m - grid) * fs) / np.nansum(fs)
        assert icis["M3_recalibrated"] < icis["M2_distorted"] / 3
        assert icis["M3_recalibrated"] <= icis["M1_logistic"] + 0.002
