#!/usr/bin/env bash
# Acquire the Diabetes 130-US hospitals dataset (UCI ML Repository ID 296)
# into data/raw/, verifying the archive SHA-256 recorded in
# data/raw/ACQUISITION_LEDGER.md. The dataset is publicly downloadable;
# see the ledger for citation and terms.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
ZIP="data/raw/dataset_diabetes.zip"
CSV="data/raw/diabetic_data.csv"
EXPECTED_ZIP_SHA256="66d9820c3bd5fd2abf7957bda2d9714b2de1e7bba1ac1f1ff8fa85e83c22cb33"
EXPECTED_CSV_SHA256="0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"

mkdir -p data/raw
if [ ! -f "$ZIP" ]; then
    echo "downloading $URL"
    curl -fsSL "$URL" -o "$ZIP"
fi
echo "$EXPECTED_ZIP_SHA256  $ZIP" | sha256sum -c -
unzip -o "$ZIP" -d data/raw
# the archive wraps its files in a dataset_diabetes/ subdirectory
[ -f data/raw/dataset_diabetes/diabetic_data.csv ] && mv -f data/raw/dataset_diabetes/diabetic_data.csv data/raw/
echo "$EXPECTED_CSV_SHA256  $CSV" | sha256sum -c -
echo "data acquisition OK: $CSV"
