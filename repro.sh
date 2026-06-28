#!/usr/bin/env bash
# ============================================================================
# EndoSeg — one-command reproduction (STUB / commented outline).
#
# Goal: a fresh clone + this script reproduces training and the browser demo.
# Fill in each step as the pipeline is implemented. Keep it idempotent and
# pin every version for reproducibility (PRD §8).
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "==> EndoSeg repro :: root = ${ROOT}"

# ----------------------------------------------------------------------------
# 0. Prereqs
#    - Docker installed; (optional) NVIDIA GPU + container toolkit for training.
#    - Nebius account + Serverless access + CLI configured (for cloud steps).
#    - Python 3.11 if running steps locally instead of in Docker.
# ----------------------------------------------------------------------------
# command -v docker >/dev/null || { echo "docker required"; exit 1; }

# ----------------------------------------------------------------------------
# 1. Get data (verify MMOTU license first!)
# ----------------------------------------------------------------------------
# MMOTU_ARCHIVE_URL=<url> bash data/download_mmotu.sh

# ----------------------------------------------------------------------------
# 2. Preprocess  (locally or as Nebius Job N1 — nebius/job_preprocess.yaml)
# ----------------------------------------------------------------------------
# python jobs/preprocess/preprocess.py \
#     --raw-dir data/raw/mmotu --out-dir data/processed --img-size 256 --clahe

# ----------------------------------------------------------------------------
# 3. Fine-tune the U-Net  (GPU; locally or as Nebius Job N2 — job_finetune.yaml)
# ----------------------------------------------------------------------------
# python jobs/finetune/train_unet.py \
#     --data-dir data/processed --out-dir checkpoints --epochs 50

# ----------------------------------------------------------------------------
# 4. Export + quantize to ONNX  (Nebius Job N3)
# ----------------------------------------------------------------------------
# python jobs/export/export_onnx.py \
#     --checkpoint checkpoints/unet_best.pt --out-dir browser/models --quantize int8

# ----------------------------------------------------------------------------
# 5. Verify ONNX parity
# ----------------------------------------------------------------------------
# pytest tests/test_onnx_parity.py

# ----------------------------------------------------------------------------
# 6. (Optional) Deploy the Endpoint + proxy
# ----------------------------------------------------------------------------
# nebius serverless endpoint create -f nebius/endpoint.yaml
# NEBIUS_TOKEN=*** NEBIUS_ENDPOINT_URL=<url> python proxy/proxy.py

# ----------------------------------------------------------------------------
# 7. Serve the browser demo (static)
# ----------------------------------------------------------------------------
# python -m http.server 8000 --directory browser
# echo "open http://localhost:8000"

echo "==> repro.sh is a stub. Uncomment steps as they are implemented."
