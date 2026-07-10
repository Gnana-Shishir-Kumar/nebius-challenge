# EndoSeg

> **Fine-tuned in the cloud. Inferred privately in your browser — your scan never leaves your laptop.**

Browser-native gynecological (ovarian-lesion) ultrasound segmentation built for the
**Nebius Serverless AI Builders Challenge** — Healthcare & Life Sciences track.
`#NebiusServerlessChallenge`

> ⚠️ **Research/education only.** EndoSeg is not a medical device and makes no
> diagnostic claims. It performs ovarian-lesion segmentation and can *flag*
> endometriomas — it does **not** diagnose endometriosis.

---

## Overview

EndoSeg is a complete cloud-to-browser medical image segmentation pipeline. A compact
2D U-Net is fine-tuned on the MMOTU ovarian-tumor ultrasound dataset using **Nebius
Serverless Jobs** — one job for preprocessing, one for GPU training, and one for ONNX
export. The exported model is quantized to under 50 MB and shipped inside a static
webpage where all inference runs locally in the browser using ONNX Runtime Web and
WebGPU. No server, no upload, no waiting.

The privacy-by-design angle is the distinguishing feature. Medical images are sensitive
by nature; most cloud demos require users to upload their scans. EndoSeg inverts this
model: the trained weights are fetched once and cached, then every inference happens
entirely on the user's device using client-side JavaScript. A second, heavier
foundation-model fine-tune lives on a **Nebius Endpoint** for a "compare to cloud"
feature — but this path is opt-in, routed through a token-hiding proxy, and clearly
labelled so users always know where their data goes.

---

## Architecture

```
PUBLIC DATA (MMOTU ovarian-tumor ultrasound)
          │
          ▼
    Nebius Object Storage
   (raw  /  processed  /  checkpoints)
      │           │              │
      ▼           ▼              ▼
  Job N1      Job N2         Job N3
(Preprocess) (Fine-tune)  (Eval + ONNX export)
  CPU only    1× L40s GPU    CPU only
      │           │              │
      └─────── checkpoints ──────┘
                                 │
                        unet_best.pth
                           │           │
                   quantized ONNX   full-precision
                   (browser/model/) checkpoint
                           │              │
                    Browser App      Endpoint N4
                 (ONNX Runtime Web)  (FastAPI + ORT)
                  WebGPU / WASM ↑        ▲
                                └──proxy─┘
                             (token hidden; opt-in)
```

**Two-model split:** a compact U-Net runs entirely in the browser via ONNX Runtime Web
(quantized, ≤ 50 MB, 256×256, WebGPU with WASM fallback). A heavier foundation-model
fine-tune (UltraSam / MedSAM2) lives on the Nebius Endpoint for the optional
"compare to cloud" feature — higher accuracy at the cost of a round-trip through the
token-hiding proxy.

### Repository layout

```
endoseg/
  jobs/preprocess/    N1: resize, normalize, patient-disjoint split
  jobs/finetune/      N2: U-Net training loop (Dice+BCE, Albumentations)
  jobs/export/        N3: PyTorch → ONNX → fp16/int8 + parity check
  endpoint/           N4: FastAPI /predict JSON API (ORT, CUDA/CPU)
  proxy/              N7: token-hiding proxy (CORS, Bearer injection)
  models/             unet.py · losses.py · metrics.py (shared package)
  browser/            static ONNX demo (HTML · CSS · JS · Web Worker)
  nebius/             Job + Endpoint YAML configs
  data/               dataset download, license notes, EDA script
  tests/              ONNX parity pytest
  notebooks/          EDA notebook
  blog/               submission blog draft
  repro.sh            one-command reproduction
```

---

## Dataset

**MMOTU** (Multi-Modality Ovarian Tumor Ultrasound) — 1 469 2D ultrasound images with
pixel-level segmentation masks across **8 ovarian-tumor classes**:

| ID | Class |
|----|-------|
| 0 | Chocolate cyst (endometrioma) |
| 1 | Serous cystadenoma |
| 2 | Teratoma |
| 3 | Theca cell tumor |
| 4 | Simple cyst |
| 5 | Normal ovary |
| 6 | Mucinous cystadenoma |
| 7 | High-grade serous cystadenoma |

Paper: [arXiv:2207.06799](https://arxiv.org/abs/2207.06799) —
*MMOTU: A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised
Cross-Domain Semantic Segmentation.*
GitHub: <https://github.com/cv516Buaa/MMOTU_DS2Net>

**License:** Verify the MMOTU license permits research use and derived-model
redistribution before building on it. See [`data/README.md`](data/README.md) for
citation template and license checklist.

```bash
# Images are hosted on Google Drive (not in the git repo).
# Option A — Google Drive ID (needs pip install gdown):
MMOTU_GDRIVE_ID=<file-id> bash data/download_mmotu.sh

# Option B — direct archive URL:
MMOTU_ARCHIVE_URL=<url> bash data/download_mmotu.sh

# Option C — manual: unzip OTU_2d so that data/MMOTU_DataSet/{images,annotations} exists.
```

---

## Setup

**Prerequisites:** Python 3.11+, Docker (for Nebius jobs), Node.js / `npx` (optional,
for serving the browser demo).

```bash
# 1. Clone
git clone https://github.com/Gnana-Shishir-Kumar/nebius-challenge.git
cd nebius-challenge

# 2. Python environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download dataset (see Dataset section above for options)
bash data/download_mmotu.sh

# 4. Smoke-test the model (no data required)
python -c "from models import UNet2D; m = UNet2D.for_browser(); print(m.model_size_mb(), 'MB')"
pytest tests/                      # ONNX parity test (needs unet.onnx in browser/model/)
```

**Hardware:** CPU is sufficient for preprocessing, export, and the browser demo.
Fine-tuning expects a GPU (locally or via Nebius Job N2). The browser demo needs a
WebGPU-capable browser; it falls back to WASM automatically.

---

## Nebius Pipeline

### N1 — Preprocessing Job (CPU)

Resizes images to 256×256, applies optional CLAHE, binarizes masks
(lesion vs background), and writes a patient-disjoint 70/15/15 train/val/test split
manifest (`splits.json`) to object storage.

```bash
# Submit via Nebius CLI (set container_image and bucket names in nebius/job_preprocess.yaml first):
nebius serverless job create -f nebius/job_preprocess.yaml

# Or run locally:
python jobs/preprocess/preprocess.py \
    --raw-dir data/MMOTU_DataSet \
    --out-dir data/processed \
    --clahe
```

### N2 — Fine-tuning Job (GPU)

Trains the compact U-Net with Dice+BCE loss, Albumentations augmentation, and
ReduceLROnPlateau scheduling. Checkpoints the best model by validation Dice to
`checkpoints/unet_best.pth`. Run a GPU-free smoke test with `--smoke-test`.

```bash
nebius serverless job create -f nebius/job_finetune.yaml

# Or locally (GPU recommended):
python jobs/finetune/train_unet.py \
    --data-dir data/processed \
    --out-dir checkpoints \
    --epochs 50 --batch-size 16

# Smoke test (CPU, no data required):
python jobs/finetune/train_unet.py --smoke-test
```

### N3 — Export Job (CPU)

Loads the checkpoint, exports to ONNX (opset 17), optionally converts to fp16, runs
a parity check against PyTorch (MAE < 0.01), and writes `browser/model/unet.onnx`.

```bash
nebius serverless job create -f nebius/job_export.yaml

# Or locally:
python jobs/export/export_onnx.py --validate
```

### N4 — Endpoint

Hosts the full-precision model behind a FastAPI `/predict` JSON API. The browser
app's "compare to cloud" button calls this through the token-hiding proxy.

```bash
nebius serverless endpoint create -f nebius/endpoint.yaml

# Run endpoint locally (set MODEL_PATH to a real .onnx file):
MODEL_PATH=browser/model/unet.onnx uvicorn endpoint.serve:app --port 8000

# Run proxy (in a second terminal):
ENDPOINT_URL=http://localhost:8000 NEBIUS_TOKEN=local \
    uvicorn proxy.proxy:app --port 8080
```

### Job summary

| Job | Container | GPU | Approx runtime | Approx cost |
|-----|-----------|-----|----------------|-------------|
| N1 Preprocess | `endoseg-preprocess:latest` | None | ~10 min | < $0.01 |
| N2 Fine-tune | `endoseg-finetune:latest` | 1× L40s | ~2–4 hr | ~$3–8 |
| N3 Export | `endoseg-export:latest` | None | ~5 min | < $0.01 |
| N4 Endpoint | `endoseg-endpoint:latest` | 1× L40s | on-demand | ~$0.03/req |

> Before submitting jobs, fill in the container registry URL (project ID) and the two
> object-storage bucket names in each YAML — they are marked with inline comments.

---

## Browser Demo

Static site — no backend is required for local inference.

```bash
# After export_onnx.py has written browser/model/unet.onnx:

# Option A — Python built-in server:
python -m http.server 8000 --directory browser
# open http://localhost:8000

# Option B — npx serve (auto-reloads, sets correct MIME types):
npx serve browser/
```

**Features:**
- Drag-and-drop upload or click a sample image from the gallery
- One-click local inference (ONNX Runtime Web, WebGPU backend with WASM fallback)
- Red/orange mask overlay with adjustable opacity
- Latency and backend readout (WebGPU vs WASM)
- Optional "Compare to cloud" button — sends the image through the proxy to Endpoint N4
  and shows local vs cloud Dice delta

**Privacy:** The model file is fetched once and cached in the browser. After that,
inference runs 100% locally — images never leave the device. The "compare to cloud"
path is explicitly opt-in and clearly labelled.

---

## Results

Trained on real MMOTU data (1469 images, stratified random split — see Known
limitations), 50 epochs, GPU (~25 min on an RTX 3050 Ti laptop GPU).

| Model | Split | Dice | IoU | Latency |
|-------|-------|------|-----|---------|
| U-Net (ONNX, browser, fp32) | val | 0.7643 (epoch 33) | 0.6703 | – ms |
| U-Net (ONNX, browser, fp32) | test | 0.7545 | 0.6553 | – ms |
| Foundation model (Endpoint) | test | – | – | – ms |

Parity delta (browser fp32 vs Endpoint): MAE reported by `export_onnx.py --validate`.

### Known limitations

- **Small-lesion under-segmentation.** A visual sanity check on 8 random test-split
  images (`jobs/finetune/visualize_predictions.py` → `data/prediction_grid.png`) found
  per-sample Dice ranging from **0.02 to 0.97**. The model correctly localizes
  medium/large lesions (6 of 8 samples, Dice 0.63–0.97), but on the 2 worst cases it
  confidently predicts a large, well-formed blob in roughly the right neighborhood
  when the true lesion is small or thin — i.e. it isn't collapsing to all-background,
  but it appears to fall back to a learned "typical lesion" prior when the real lesion
  doesn't match that shape. Hypothesis: small/thin lesions are underrepresented in the
  1029-image training split, so the model hasn't learned to trust small foreground
  regions. Not yet mitigated — a candidate fix is loss reweighting or oversampling
  small-lesion cases.
- **Split is not truly patient-disjoint.** MMOTU filenames in this data mirror are
  plain numeric IDs with no recoverable patient/case grouping, so
  `jobs/preprocess/preprocess.py` falls back to a stratified random split by file ID
  (with an explicit warning printed at preprocessing time). There is a small risk of
  same-patient images (e.g. adjacent frames of the same scan) landing in both train
  and val/test, which could inflate the reported Dice slightly.

---

## Reproducing

```bash
bash repro.sh
```

The script runs all six steps in sequence:

| Step | Command | What it does |
|------|---------|-------------|
| 1 | `bash data/download_mmotu.sh` | Clone MMOTU repo + download image archive |
| 2 | `python jobs/preprocess/preprocess.py ...` | Resize, normalize, split |
| 3 | `python jobs/finetune/train_unet.py ...` | Train U-Net, save checkpoint |
| 4 | `python jobs/export/export_onnx.py --validate` | ONNX export + parity check |
| 5 | `npx serve browser/` | Serve static browser demo |
| 6 | *(echo)* | Pointer to Nebius YAML configs |

**Expected outputs after full local run:**
- `data/processed/splits.json` — train/val/test manifest
- `checkpoints/unet_best.pth` — best checkpoint by validation Dice
- `browser/model/unet.onnx` — exported fp32 ONNX (7.5 MB)
- Console: `Parity check passed: MAE=0.000000`

---

## Team

| Name | Role |
|------|------|
| [Your name] | ML pipeline, Nebius integration |
| [Teammate name] | Browser app, frontend |

---

## Citation

```bibtex
@misc{endoseg2026,
  title  = {EndoSeg: Browser-Native Ovarian-Lesion Ultrasound Segmentation on Nebius Serverless},
  year   = {2026},
  note   = {Nebius Serverless AI Builders Challenge, Healthcare track},
  url    = {https://github.com/Gnana-Shishir-Kumar/nebius-challenge}
}

@article{zhao2022mmotu,
  title   = {MMOTU: A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for
             Unsupervised Cross-Domain Semantic Segmentation},
  author  = {Zhao, Qi and Li, Shuchang and others},
  journal = {arXiv preprint arXiv:2207.06799},
  year    = {2022},
  url     = {https://arxiv.org/abs/2207.06799}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Dataset: MMOTU is distributed by its authors under their own terms; see
[`data/README.md`](data/README.md) for the license checklist and required citation.
This repository contains no raw dataset files.
