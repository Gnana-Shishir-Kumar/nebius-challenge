#!/usr/bin/env bash
# Download + verify the MMOTU ovarian-tumor ultrasound dataset.
#
# IMPORTANT: Verify the MMOTU license BEFORE building on it (PRD §8). Confirm it
# permits research use and redistribution of derived models. Cite the dataset.
# This script does NOT bundle data; it points at the official source.
set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")" && pwd)/raw/mmotu}"
MMOTU_URL="${MMOTU_URL:-https://github.com/cv516Buaa/MMOTU_DS2Net}"  # official repo / dataset pointer

echo "==> EndoSeg :: MMOTU download"
echo "    target dir : ${DATA_DIR}"
echo "    source     : ${MMOTU_URL}"
mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# 1. Obtain the dataset.
#    MMOTU is distributed via the authors' repo / linked cloud storage. Fill in
#    the concrete archive URL once you've confirmed the license + access method.
#
#    Example (uncomment and set MMOTU_ARCHIVE_URL):
#    curl -L "${MMOTU_ARCHIVE_URL}" -o "${DATA_DIR}/mmotu.zip"
# ---------------------------------------------------------------------------
if [[ -z "${MMOTU_ARCHIVE_URL:-}" ]]; then
  echo "!! MMOTU_ARCHIVE_URL not set."
  echo "   Visit ${MMOTU_URL}, accept the license, then re-run with:"
  echo "   MMOTU_ARCHIVE_URL=<url> bash download_mmotu.sh"
  exit 1
fi

curl -L "${MMOTU_ARCHIVE_URL}" -o "${DATA_DIR}/mmotu.zip"

# ---------------------------------------------------------------------------
# 2. Verify integrity (set EXPECTED_SHA256 once known).
# ---------------------------------------------------------------------------
if [[ -n "${EXPECTED_SHA256:-}" ]]; then
  echo "==> verifying checksum"
  echo "${EXPECTED_SHA256}  ${DATA_DIR}/mmotu.zip" | sha256sum -c -
fi

# ---------------------------------------------------------------------------
# 3. Extract into images/ and masks/ subfolders expected by preprocess.py.
# ---------------------------------------------------------------------------
echo "==> extracting"
unzip -q -o "${DATA_DIR}/mmotu.zip" -d "${DATA_DIR}"

echo "==> done. Raw data in ${DATA_DIR}"
echo "    Next: run the preprocessing Job (see nebius/job_preprocess.yaml)."
