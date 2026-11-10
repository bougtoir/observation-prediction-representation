"""Aggregate every analysis CSV into one canonical machine-readable
result registry: analysis/canonical_results.json.

Flat key space (dot-separated) so manuscript/table/figure builders and
the numeric-consistency audit can reference a single value by name:

    realdata.<metric>.<model>          e.g. realdata.auc.M4_gbm
    realdata.sens.<model>@<thr>        e.g. realdata.sens.M1_logistic@0.05
    realdata.spec.<model>@<thr>
    realdata.ppv.<model>@<thr>
    realdata.nb.<model>@<pt>           e.g. realdata.nb.M1_logistic@0.05
    realdata.kappa.<model>@<thr>
    realdata.kappa_max.<model>
    realdata.t_at_kappa_max.<model>
    realdata.t_at_nb_max.<model>
    realdata.n_test / .prevalence / .n_cohort / .n_events / .n_encounters
    realdata.recal.intercept / .slope / .k / .expected_slope
    boot.<metric>.<A>-<B>.<mean|lo|hi> e.g. boot.auc.M4_gbm-M1_logistic.lo
    sim.<scenario>.<model>.<metric>    e.g. sim.D_overlap_location.mid_overlap.auc

This file regenerates; never edit the JSON by hand.
"""
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(ROOT, "analysis")

C = {}

# --- real data: held-out metrics --------------------------------------
m = pd.read_csv(os.path.join(AN, "realdata_metrics.csv"))
for _, r in m.iterrows():
    mod = r["model"]
    for col in m.columns:
        if col == "model":
            continue
        key = col
        for pfx in ("sens", "spec", "ppv", "nb"):
            if col.startswith(pfx + "@"):
                thr = col.split("@", 1)[1]
                key = f"{pfx}.{mod}@{thr}"
                break
        else:
            key = f"{col}.{mod}"
        C[f"realdata.{key}"] = float(r[col])

# --- cohort-level counts ----------------------------------------------
co = pd.read_csv(os.path.join(AN, "realdata_cohort.csv")).iloc[0]
for col in co.index:
    C[f"realdata.{col}"] = float(co[col])

# --- kappa -------------------------------------------------------------
k = pd.read_csv(os.path.join(AN, "realdata_kappa.csv"))
for _, r in k.iterrows():
    mod = r["model"]
    C[f"realdata.kappa_max.{mod}"] = float(r["kappa_max"])
    C[f"realdata.t_at_kappa_max.{mod}"] = float(r["t_at_kappa_max"])
    C[f"realdata.t_at_nb_max.{mod}"] = float(r["t_at_nb_max"])
    for col in k.columns:
        if col.startswith("kappa@"):
            thr = col.split("@", 1)[1]
            C[f"realdata.kappa.{mod}@{thr}"] = float(r[col])

# --- recalibration fit --------------------------------------------------
rc = pd.read_csv(os.path.join(AN, "recalibration_fit.csv")).iloc[0]
C["realdata.recal.intercept"] = float(rc["intercept"])
C["realdata.recal.slope"] = float(rc["slope"])
C["realdata.recal.k"] = float(rc["k"])
C["realdata.recal.expected_slope"] = float(rc["expected_slope"])

# --- bootstrap contrasts -------------------------------------------------
b = pd.read_csv(os.path.join(AN, "realdata_bootstrap.csv"))
for _, r in b.iterrows():
    a, _, bb = r["contrast"].partition(" - ")
    C[f"boot.{r['metric']}.{a}-{bb}.mean"] = float(r["mean"])
    C[f"boot.{r['metric']}.{a}-{bb}.lo"] = float(r["lo"])
    C[f"boot.{r['metric']}.{a}-{bb}.hi"] = float(r["hi"])

# --- simulation summary ---------------------------------------------------
s = pd.read_csv(os.path.join(AN, "simulation_summary.csv"))
for _, r in s.iterrows():
    for col in s.columns:
        if col in ("scenario", "model"):
            continue
        C[f"sim.{r['scenario']}.{r['model']}.{col}"] = float(r[col])

with open(os.path.join(AN, "canonical_results.json"), "w") as f:
    json.dump(C, f, indent=1, sort_keys=True)
print(f"canonical_results.json: {len(C)} keys")
