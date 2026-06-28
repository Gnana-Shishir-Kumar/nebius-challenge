# EndoSeg

> **Fine-tuned in the cloud. Inferred privately in your browser — your scan never leaves your laptop.**

Browser-native gynecological (ovarian-lesion) ultrasound segmentation. Models are
fine-tuned on **Nebius Serverless** (Jobs + Endpoint) and a compact U-Net runs
**fully client-side** in the browser via ONNX Runtime Web + WebGPU.

> ⚠️ **Research/education only.** EndoSeg is not a medical device and makes no
> diagnostic claims. It performs ovarian-lesion ultrasound segmentation and can
> *flag* endometriomas — it does **not** diagnose endometriosis.

Built for the **Nebius Serverless AI Builders Challenge** — Healthcare & Life
Sciences track. `#NebiusServerlessChallenge`

---

## Overview

EndoSeg combines three things that rarely appear together:

1. A real medical segmentation fine-tuning pipeline on **Nebius Serverless Jobs**
   (preprocess → fine-tune → eval/ONNX export).
2. **Client-side, in-browser inference** so medical images never leave the device.
3. A hosted **Nebius Endpoint** for full-precision cloud inference and a
   "compare to cloud" feature, reached via a thin token-hiding proxy.

**Two-model split:**
- **Browser model** — compact 2D U-Net, ONNX-clean, quantized (≤ ~50 MB), 256×256.
- **Endpoint model** — heavier foundation-model fine-tune (UltraSam / MedSAM2 / SAM2).

### Repository layout

```
endoseg/
  jobs/          preprocess · finetune · export   (Nebius Jobs N1–N3)
  endpoint/      FastAPI inference server          (Nebius Endpoint N4)
  proxy/         token-hiding proxy                (N7)
  models/        unet.py · losses.py · metrics.py
  browser/       static client-side ONNX demo
  notebooks/     EDA
  nebius/        Job / Endpoint YAML configs
  data/          dataset download + license notes
  blog/          submission blog draft
  tests/         ONNX parity test
  repro.sh       one-command reproduction (stub)
```

## Setup

```bash
# 1. Clone
git clone https://github.com/your-org/endoseg.git && cd endoseg

# 2. Python env (3.11)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Docker
docker build -t endoseg .

# 4. Sanity check the model
python models/unet.py            # prints output shape + param count
pytest tests/test_onnx_parity.py # PyTorch vs ONNX parity
```

**Hardware:** CPU is fine for preprocessing/export/tests. Fine-tuning expects a
GPU (locally or via the Nebius fine-tune Job). Browser demo needs a WebGPU-capable
browser (falls back to WASM).

## Dataset

Public, anonymized data only — **MMOTU** ovarian-tumor ultrasound (MVP) and
optionally **GLENDA** laparoscopy (stretch). See [`data/README.md`](data/README.md).

```bash
# Verify the MMOTU license first, then:
MMOTU_ARCHIVE_URL=<url> bash data/download_mmotu.sh
```

> ⚠️ **Verify MMOTU's license before building on it** (research use + derived-model
> redistribution). Cite all datasets. GLENDA is typically CC BY-NC / research-only.

## Nebius Pipeline

| Stage | Component | Config |
|---|---|---|
| N1 Preprocess (CPU) | `jobs/preprocess/` | `nebius/job_preprocess.yaml` |
| N2 Fine-tune (GPU) | `jobs/finetune/` | `nebius/job_finetune.yaml` |
| N3 Eval + ONNX export | `jobs/export/` | (run as a Job variant) |
| N4 Endpoint inference | `endpoint/` | `nebius/endpoint.yaml` |
| N7 Token-hiding proxy | `proxy/` | — |

```bash
# Build + push component images, then create Jobs / Endpoint (CLI keys illustrative)
nebius serverless job create     -f nebius/job_preprocess.yaml
nebius serverless job create     -f nebius/job_finetune.yaml
nebius serverless endpoint create -f nebius/endpoint.yaml
```

Object storage holds raw data, processed data, and checkpoints. Capture Job logs,
GPU utilization, cost, and the Endpoint URL as proof-of-execution for submission.

## Browser Demo

Static site — no backend needed for local inference.

```bash
# After exporting a model to browser/models/unet_int8.onnx:
python -m http.server 8000 --directory browser
# open http://localhost:8000
```

Features: drag-and-drop / sample gallery, ONNX Runtime Web (WebGPU + WASM
fallback), mask overlay with opacity slider, backend + latency readout, Web
Worker inference, and an optional **"Compare to cloud"** call through the proxy.
The privacy banner reflects the design: *your image never leaves this device.*

## Reproducing

```bash
bash repro.sh   # commented outline of the full pipeline; uncomment steps as implemented
```

The script walks: download → preprocess → fine-tune → export/quantize → parity
test → (optional) Endpoint + proxy → serve browser demo.

## Citation

```bibtex
@misc{endoseg2026,
  title  = {EndoSeg: Browser-Native Ovarian-Lesion Ultrasound Segmentation},
  year   = {2026},
  note   = {Nebius Serverless AI Builders Challenge submission}
}

@article{mmotu,
  title  = {MMOTU: Multi-Modal Ovarian Tumor Ultrasound dataset},
  note   = {Add full citation + confirmed license before submission}
}
```

---

*MIT licensed (see [LICENSE](LICENSE)). Research/education use only.*
