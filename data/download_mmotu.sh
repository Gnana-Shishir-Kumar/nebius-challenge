#!/usr/bin/env bash
# =============================================================================
# EndoSeg :: MMOTU dataset downloader + verifier
# -----------------------------------------------------------------------------
# Obtains the MMOTU (Multi-Modality Ovarian Tumor Ultrasound) dataset, then:
#   1. Clones / downloads MMOTU (code repo + OTU_2d image/mask data)
#   2. Verifies expected file counts (~1469 2D images + pixel-level masks)
#   3. Prints a summary: class names, per-class image counts, mask format
#   4. Flags corrupted files (zero-byte check)
#
# IMPORTANT: Verify the MMOTU license BEFORE building on it (see PRD §8 / Data
# & Datasets). Confirm it permits research use and redistribution of derived
# models, and cite the dataset. This script does NOT bundle data; it points at
# the official source and downloads on demand.
#
# The code repo is public on GitHub, but the actual images live on Google Drive
# (too large for git). You therefore typically need ONE of:
#   * MMOTU_GDRIVE_ID    - Google Drive file id for OTU_2d.zip  (needs `gdown`)
#   * MMOTU_ARCHIVE_URL  - any direct URL to an OTU_2d archive  (uses `curl`)
#   * a manual placement of the data under $DATA_DIR (script still verifies it)
#
# Usage:
#   bash data/download_mmotu.sh
#   MMOTU_GDRIVE_ID=<id>      bash data/download_mmotu.sh
#   MMOTU_ARCHIVE_URL=<url>   bash data/download_mmotu.sh
#   DATA_DIR=/custom/path     bash data/download_mmotu.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via environment variables)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/raw/mmotu}"
# Canonical public repo. NOTE: the often-cited URL
# https://github.com/cv516Buaa/MMOTU_DataSet returns 404; the live repo is:
REPO_URL="${MMOTU_REPO_URL:-https://github.com/cv516Buaa/MMOTU_DS2Net}"
REPO_DIR="${DATA_DIR}/MMOTU_DS2Net"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-1469}"

# MMOTU OTU_2d has 8 ovarian-tumor categories (id -> name), per the paper
# (arXiv:2207.06799). Index == class id used in the *_cls.txt label files.
CLASS_NAMES=(
  "Chocolate cyst (endometrioma)"  # 0
  "Serous cystadenoma"             # 1
  "Teratoma"                       # 2
  "Theca cell tumor"               # 3
  "Simple cyst"                    # 4
  "Normal ovary"                   # 5
  "Mucinous cystadenoma"           # 6
  "High grade serous cystadenoma"  # 7
)

# Colored-ish status helpers (plain text, no deps).
log()  { printf '==> %s\n' "$*"; }
warn() { printf '!!  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

log "EndoSeg :: MMOTU download + verify"
log "target dir : ${DATA_DIR}"
log "source     : ${REPO_URL}"
mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# 1. Acquire the dataset.
# ---------------------------------------------------------------------------

# 1a. Clone (or update) the code repo. This gives the documented layout, label
#     files and split lists. It does NOT include the large image set.
if command -v git >/dev/null 2>&1; then
  if [[ -d "${REPO_DIR}/.git" ]]; then
    log "repo already cloned -> ${REPO_DIR} (pulling latest)"
    git -C "${REPO_DIR}" pull --ff-only --quiet || warn "git pull failed; using existing clone"
  else
    log "cloning ${REPO_URL}"
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}" || warn "git clone failed (continuing; will still verify any local data)"
  fi
else
  warn "git not found; skipping repo clone (install git to fetch code + label files)"
fi

# 1b. Obtain the OTU_2d image/mask payload if not already present.
#     We look for an existing OTU_2d folder anywhere under DATA_DIR first.
find_otu2d() {
  # Print the first directory that contains both an images/ and an
  # annotations/ (or masks/) subdir.
  local d
  while IFS= read -r d; do
    if [[ -d "${d}/images" ]] && { [[ -d "${d}/annotations" ]] || [[ -d "${d}/masks" ]]; }; then
      printf '%s\n' "${d}"
      return 0
    fi
  done < <(find "${DATA_DIR}" -type d \( -iname 'OTU_2d' -o -iname 'OTU2d' -o -iname 'mmotu*' \) 2>/dev/null)
  return 1
}

ARCHIVE="${DATA_DIR}/OTU_2d.zip"
if ! OTU2D_DIR="$(find_otu2d)"; then
  if [[ -n "${MMOTU_ARCHIVE_URL:-}" ]]; then
    log "downloading archive via curl"
    curl -L --fail "${MMOTU_ARCHIVE_URL}" -o "${ARCHIVE}"
  elif [[ -n "${MMOTU_GDRIVE_ID:-}" ]]; then
    if command -v gdown >/dev/null 2>&1; then
      log "downloading OTU_2d from Google Drive via gdown (${MMOTU_GDRIVE_ID})"
      gdown --id "${MMOTU_GDRIVE_ID}" -O "${ARCHIVE}"
    else
      die "gdown not installed. Run: pip install gdown   (or set MMOTU_ARCHIVE_URL)"
    fi
  fi

  # Optional checksum verification of the downloaded archive.
  if [[ -f "${ARCHIVE}" && -n "${EXPECTED_SHA256:-}" ]]; then
    log "verifying archive checksum"
    echo "${EXPECTED_SHA256}  ${ARCHIVE}" | sha256sum -c -
  fi

  # Extract whatever we managed to download.
  if [[ -f "${ARCHIVE}" ]]; then
    log "extracting ${ARCHIVE}"
    if command -v unzip >/dev/null 2>&1; then
      unzip -q -o "${ARCHIVE}" -d "${DATA_DIR}"
    else
      die "unzip not found; please install it or extract ${ARCHIVE} manually"
    fi
    OTU2D_DIR="$(find_otu2d || true)"
  fi
fi

if [[ -z "${OTU2D_DIR:-}" ]] || [[ ! -d "${OTU2D_DIR:-/nonexistent}" ]]; then
  warn "Could not locate an OTU_2d dataset folder under ${DATA_DIR}."
  warn "The GitHub repo only ships code; download the images from the Google"
  warn "Drive link in ${REPO_DIR}/README.md, then re-run with one of:"
  warn "  MMOTU_GDRIVE_ID=<file-id>   bash data/download_mmotu.sh"
  warn "  MMOTU_ARCHIVE_URL=<url>     bash data/download_mmotu.sh"
  warn "or unzip OTU_2d so that ${DATA_DIR}/OTU_2d/{images,annotations} exists."
  exit 1
fi

log "using dataset root: ${OTU2D_DIR}"
IMAGES_DIR="${OTU2D_DIR}/images"
if [[ -d "${OTU2D_DIR}/annotations" ]]; then
  MASKS_DIR="${OTU2D_DIR}/annotations"
else
  MASKS_DIR="${OTU2D_DIR}/masks"
fi

# ---------------------------------------------------------------------------
# 2. Verify expected file counts.
# ---------------------------------------------------------------------------
log "verifying file counts"
n_images=$(find "${IMAGES_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
n_masks=$(find "${MASKS_DIR}"  -type f \( -iname '*.png' -o -iname '*.bmp' \)                    2>/dev/null | wc -l | tr -d ' ')

printf '    images : %s  (in %s)\n' "${n_images}" "${IMAGES_DIR}"
printf '    masks  : %s  (in %s)\n' "${n_masks}"  "${MASKS_DIR}"
printf '    expect : ~%s images + matching masks\n' "${EXPECTED_IMAGES}"

if [[ "${n_images}" -eq "${EXPECTED_IMAGES}" ]]; then
  log "image count matches expected (${EXPECTED_IMAGES})."
else
  warn "image count ${n_images} != expected ${EXPECTED_IMAGES} (proceeding; dataset releases vary)."
fi
if [[ "${n_images}" -ne "${n_masks}" ]]; then
  warn "image/mask count mismatch (${n_images} vs ${n_masks}); some pairs may be missing."
fi

# ---------------------------------------------------------------------------
# 3. Summary: class names, per-class image counts, mask format.
# ---------------------------------------------------------------------------
log "class summary (8 ovarian-tumor categories)"
for i in "${!CLASS_NAMES[@]}"; do
  printf '    %d : %s\n' "${i}" "${CLASS_NAMES[$i]}"
done

# Per-class counts come from the global-wise label files (*_cls.txt), each line
# of which looks like "<image>.JPG <classid>". Aggregate across any we find.
mapfile -t CLS_FILES < <(find "${OTU2D_DIR}" -maxdepth 2 -type f -iname '*cls*.txt' 2>/dev/null)
if [[ "${#CLS_FILES[@]}" -gt 0 ]]; then
  log "per-class image counts (from ${#CLS_FILES[@]} label file(s))"
  # Analytics only: never let a no-match grep / empty file abort the script
  # (set -euo pipefail). awk strips any stray CR so CRLF files parse cleanly.
  { cat "${CLS_FILES[@]}" 2>/dev/null \
      | awk 'NF { gsub(/\r/, ""); print $NF }' \
      | grep -E '^[0-7]$' \
      | sort \
      | uniq -c \
      | while read -r count cid; do
          printf '    class %s (%-30s): %s\n' "${cid}" "${CLASS_NAMES[$cid]}" "${count}"
        done; } || true
else
  warn "no *_cls.txt label files found; per-class counts unavailable."
  warn "(global category labels ship with the official OTU_2d release.)"
fi

# Mask format: report container/type + dimensions of one sample.
log "mask format"
sample_mask="$(find "${MASKS_DIR}" -type f \( -iname '*.png' -o -iname '*.bmp' \) 2>/dev/null | head -n 1 || true)"
if [[ -n "${sample_mask}" ]]; then
  printf '    sample      : %s\n' "${sample_mask}"
  if command -v file >/dev/null 2>&1; then
    printf '    type        : %s\n' "$(file -b "${sample_mask}")"
  fi
  if command -v identify >/dev/null 2>&1; then
    printf '    geometry    : %s\n' "$(identify -format '%wx%h %m %z-bit %[colorspace]' "${sample_mask}" 2>/dev/null || true)"
  fi
  printf '    note        : pixel-level masks; 0 = background, >0 = lesion (binarize with mask>0).\n'
else
  warn "no mask files found to inspect format."
fi

# ---------------------------------------------------------------------------
# 4. Corruption check (zero-byte files).
# ---------------------------------------------------------------------------
log "checking for corrupted (zero-byte) files"
zero_count=0
while IFS= read -r f; do
  printf '    ZERO-BYTE: %s\n' "${f}"
  zero_count=$((zero_count + 1))
done < <(find "${IMAGES_DIR}" "${MASKS_DIR}" -type f -size 0 2>/dev/null)

if [[ "${zero_count}" -eq 0 ]]; then
  log "no zero-byte files detected."
else
  warn "${zero_count} zero-byte file(s) detected above — re-download recommended."
fi

# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------
log "done."
printf '    Dataset root : %s\n' "${OTU2D_DIR}"
printf '    Images       : %s\n' "${IMAGES_DIR}"
printf '    Masks        : %s\n' "${MASKS_DIR}"
printf '    Next         : python data/explore_mmotu.py --root "%s"\n' "${OTU2D_DIR}"
printf '                   then run the preprocessing Job (nebius/job_preprocess.yaml).\n'

# Non-zero exit if we found corruption, so CI / callers can react.
[[ "${zero_count}" -eq 0 ]] || exit 2
