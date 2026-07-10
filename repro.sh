#!/usr/bin/env bash
# =============================================================================
# EndoSeg — one-command reproduction
#
# Runs the full pipeline: download → preprocess → fine-tune → ONNX export
# → browser demo. Requires Python 3.11+ and pip install -r requirements.txt.
# For Nebius cloud steps see nebius/ YAML configs.
#
# Usage:
#   bash repro.sh
#
# Step 1 requires MMOTU data. Set one of:
#   MMOTU_GDRIVE_ID=<file-id>   bash repro.sh
#   MMOTU_ARCHIVE_URL=<url>     bash repro.sh
# Or manually place data so that data/MMOTU_DataSet/{images,annotations} exists.
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "==> EndoSeg repro :: $(pwd)"

# =============================================================================
echo "=== Step 1: Download MMOTU dataset ==="
# =============================================================================
bash data/download_mmotu.sh

# =============================================================================
echo "=== Step 2: Preprocess — resize 256x256, CLAHE, patient-disjoint split ==="
# =============================================================================
python jobs/preprocess/preprocess.py \
    --raw-dir data/MMOTU_DataSet \
    --out-dir data/processed \
    --clahe

# =============================================================================
echo "=== Step 3: Fine-tune U-Net (GPU recommended; ~2-4 hr on L40s) ==="
# =============================================================================
python jobs/finetune/train_unet.py \
    --data-dir data/processed \
    --out-dir checkpoints \
    --epochs 50 \
    --batch-size 8

# =============================================================================
echo "=== Step 4: Export to ONNX and validate PyTorch/ORT parity ==="
# =============================================================================
python jobs/export/export_onnx.py --validate

# =============================================================================
echo "=== Step 5: Run browser demo ==="
# =============================================================================
echo "Open browser/index.html or run: npx serve browser/"

# =============================================================================
echo "=== Step 6: Nebius cloud deployment ==="
# =============================================================================
echo "For Nebius: see nebius/ directory for Job and Endpoint configs"
echo "  nebius serverless job create -f nebius/job_preprocess.yaml"
echo "  nebius serverless job create -f nebius/job_finetune.yaml"
echo "  nebius serverless job create -f nebius/job_export.yaml"
echo "  nebius serverless endpoint create -f nebius/endpoint.yaml"

echo "==> Done. Expected outputs:"
echo "    data/processed/splits.json"
echo "    checkpoints/unet_best.pth"
echo "    browser/model/unet.onnx"
