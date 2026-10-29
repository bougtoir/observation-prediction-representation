"""Numeric consistency audit: every numeric literal in the manuscripts is
matched against the canonical results registry.

Writes analysis/NUMERICAL_PROVENANCE_MAP.csv and docs/consistency report
data. Numbers not found in the registry are flagged REVIEW (they are
either legitimate constants — seeds, thresholds, grids — or stale).
"""
import os, re, json, csv, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = json.load(open(os.path.join(ROOT, "analysis/canonical_results.json")))

# value -> keys index with rounding tolerance
def build_index():
    idx = []
    for k, v in CANON.items():
        try:
            idx.append((float(v), k))
        except (TypeError, ValueError):
            pass
    return idx

IDX = build_index()

ALLOWED = {  # legitimate constants (grids, seeds, hyperparams, section/table numbers)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "100", "20000", "150", "0.02", "0.2", "2026", "1956", "1973",
    "296", "130", "71518", "2014", "1999", "2008", "20261006", "200,500,100", "5000,200", "1000", "200500100", "5000200",
}

def find(num):
    for cand in (num, num / 100.0):  # allow percent-vs-fraction forms
        for v, k in IDX:
            for tol in (0, 0.0005, 0.005, 0.05):
                if abs(v - cand) <= tol:
                    return k, v, tol
    return None


def scan(path, tag):
    text = open(path).read()
    # strip latex commands so \approx0.72 scans as 0.72
    text = re.sub(r"\\[a-zA-Z]+\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    rows = []
    for m in re.finditer(r"(?<![\w{}.])(\d+\.?\d*(?:,\d{3})+|\d+\.\d+|\d+)", text):
        lit = m.group(0).replace(",", "")
        if lit in ALLOWED or lit.startswith("0") and len(lit) <= 2:
            continue
        try:
            num = float(lit)
        except ValueError:
            continue
        line = text[:m.start()].count("\n") + 1
        ctx = text[max(0, m.start()-40):m.end()+40].replace("\n", " ")
        hit = find(num)
        if hit:
            rows.append([tag, line, m.group(0), hit[0], f"{hit[1]:.6g}",
                         "VERIFIED", ctx])
        else:
            rows.append([tag, line, m.group(0), "", "", "REVIEW", ctx])
    return rows


def main():
    out = os.path.join(ROOT, "analysis/NUMERICAL_PROVENANCE_MAP.csv")
    rows = scan(os.path.join(ROOT, "manuscript/manuscript_text.md"), "SiM")
    rows += scan(os.path.join(ROOT, "jce/manuscript_text.md"), "JCE")
    rows += scan(os.path.join(ROOT, "supplement/supplement_text.md"), "Supp")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["manuscript_location", "result_key", "canonical_file",
                    "canonical_value", "rendered_value", "status", "context"])
        for tag, line, lit, key, val, status, ctx in rows:
            w.writerow([f"{tag}:line{line}", key,
                        "analysis/canonical_results.json" if key else "",
                        val, lit, status, ctx])
    rev = [r for r in rows if r[5] == "REVIEW"]
    print(f"{len(rows)} literals; {len(rows)-len(rev)} verified, {len(rev)} REVIEW")
    for r in rev:
        print("  REVIEW", r[0], "line", r[1], ":", r[2], "|", r[6][:100])


if __name__ == "__main__":
    main()
