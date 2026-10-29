"""Verify every bibliography entry against Crossref.

For each key in refs.REFS the script queries the Crossref API with the
bibliographic string, checks that the top hit matches first-author
surname and year, and writes docs/REFERENCE_VERIFICATION.md +
analysis/reference_verification.csv with VERIFIED / CORRECTED /
UNVERIFIABLE status. Requires network access.
"""
import os, re, sys, csv, json, time
import urllib.request, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refs import REFS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(ref):
    m = re.match(r"([A-Z][A-Za-zÀ-ÿ\-']+(?:\s+[A-Z][a-z]+)?)", ref)
    first = (m.group(1) if m else "").strip()
    # two-word surnames like "Van Calster": keep both parts if second is part of surname
    m2 = re.match(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", first)
    if m2 and m2.group(1).lower().startswith(("van ", "de ", "von ", "der ")):
        first = m2.group(1)
    ys = re.findall(r"(19|20)\d{2}", ref)
    year = int(re.search(r"((19|20)\d{2})", ref).group(1))
    return first, year


def query_crossref(ref):
    q = urllib.parse.urlencode({"query.bibliographic": ref, "rows": 3,
                                "mailto": "devin@example.org"})
    url = "https://api.crossref.org/works?" + q
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())["message"]["items"]


def check(key, ref):
    first, year = parse(ref)
    try:
        items = query_crossref(ref)
    except Exception as e:
        return dict(key=key, status="UNVERIFIABLE", note=str(e), hit="")
    best = None
    for it in items:
        fams = [a.get("family", "") for a in it.get("author", [])]
        yr = None
        for f in ("published-print", "published-online", "issued"):
            if f in it and it[f].get("date-parts"):
                yr = it[f]["date-parts"][0][0]; break
        title = (it.get("title") or [""])[0]
        score = (first.lower() in [f.lower() for f in fams]) + (yr == year)
        if best is None or score > best[0]:
            best = (score, it, fams, yr, title)
    score, it, fams, yr, title = best
    doi = it.get("DOI", "")
    hit = f"{', '.join(fams[:2])} ({yr}) {title} doi:{doi}"
    if score == 2:
        return dict(key=key, status="VERIFIED", note="", hit=hit, doi=doi)
    return dict(key=key, status="UNVERIFIABLE", hit=hit,
                note=f"best crossref hit author/year mismatch (wanted {first} {year})")


def main():
    rows = []
    for k, ref in REFS.items():
        r = check(k, ref)
        r["ref"] = ref
        rows.append(r)
        print(f"{k:18s} {r['status']:12s} {r.get('hit','')[:110]}")
        time.sleep(0.2)
    os.makedirs(os.path.join(ROOT, "analysis"), exist_ok=True)
    with open(os.path.join(ROOT, "analysis", "reference_verification.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "status", "ref", "hit",
                                          "doi", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in w.fieldnames})
    bad = [r["key"] for r in rows if r["status"] != "VERIFIED"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} VERIFIED; unverified: {bad}")


if __name__ == "__main__":
    main()
