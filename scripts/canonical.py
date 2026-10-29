"""Accessor for the canonical result registry.

Builders and audits must read every analysis-derived number through
`canon(key)` / `canon_fmt(key, spec)` — never retype result values.
Run `scripts/make_canonical.py` after the analysis pipeline.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(ROOT, "analysis", "canonical_results.json")
_C = None


def _load():
    global _C
    if _C is None:
        with open(_PATH) as f:
            _C = json.load(f)
    return _C


def canon(key):
    c = _load()
    if key not in c:
        raise KeyError(f"canonical result {key!r} missing from {_PATH} "
                       f"(run scripts/make_canonical.py)")
    return c[key]


def canon_fmt(key, spec):
    """Return the canonical value formatted with a format spec, e.g.
    canon_fmt('realdata.auc.M1_logistic', '.3f')."""
    return format(canon(key), spec)
