# EndoSeg: Fine-tuned in the Cloud, Inferred Privately in Your Browser

Endometriosis affects an estimated **190 million** people worldwide, yet the condition remains difficult to diagnose early. Ultrasound is often the first imaging step, but ML tooling that clinicians and researchers can actually *try* — without shipping patient data to a third-party API — is still surprisingly scarce. **EndoSeg** is our answer for the [Nebius Serverless AI Builders Challenge](https://challenge.academy.nebius.com/) (#NebiusServerlessChallenge): a reproducible pipeline that fine-tunes an ovarian-lesion segmentation model on **Nebius Serverless Jobs**, serves a heavier checkpoint on a **Nebius Endpoint**, and runs a compact **U-Net entirely in the browser** so a user's scan never has to leave their laptop.

> **Disclaimer:** EndoSeg is for research and education only. It is not a medical device and makes no diagnostic claims. It performs ovarian-lesion ultrasound segmentation and may *flag* suspicious regions — it does **not** diagnose endometriosis.

## Dataset

For the MVP we target the **MMOTU** multi-modal ovarian tumor ultrasound dataset: public, anonymized B-mode frames with lesion annotations suitable for binary segmentation. Before building on MMOTU you must verify its license for research use and derived-model redistribution. We apply a patient-disjoint train/validation/test split, resize frames to **256×256**, optional CLAHE contrast normalization, and store processed tensors on Nebius object storage for training Jobs.

A stretch goal is **GLENDA** laparoscopy imagery for a second track; that dataset is typically CC BY-NC and research-only. All challenge submissions must use publicly available data — no patient-identifiable uploads in the demo.

## Architecture Choices

We deliberately split the system into **two models**:

1. **Browser U-Net** — a small 2D U-Net exported to ONNX, quantized to roughly **30–50 MB**, input shape `(1, 1, 256, 256)`. It is ONNX-clean (no exotic post-processing ops), maps well to **ONNX Runtime Web** with WebGPU or WASM, and fits the privacy story: inference happens in a Web Worker off the UI thread.

2. **Endpoint foundation fine-tune** — a heavier model (UltraSAM / MedSAM2 / SAM2 class) for full-precision cloud inference and the optional **Compare to cloud** feature. Foundation models offer better boundary fidelity but are too large and operationally complex for in-browser deployment today; many SAM-style post-processing steps do not translate cleanly to WebGPU.

This split lets us optimize for **latency and privacy** locally while still demonstrating **Nebius Serverless Endpoints** for batch-quality inference when the user opts in.

## The Nebius Pipeline

The cloud side is four stages plus a token-hiding proxy:

| Stage | Component | Role |
|---|---|---|
| N1 | Preprocess Job (CPU) | MMOTU ETL → object storage |
| N2 | Fine-tune Job (GPU) | Train U-Net, log Dice/IoU, checkpoint |
| N3 | Export Job | ONNX export + int8 quantize + parity test |
| N4 | Endpoint | FastAPI `POST /infer` on GPU, scale-to-zero |
| N7 | Proxy | Injects Nebius bearer token server-side |

Illustrative Job config (preprocess):

```yaml
# nebius/job_preprocess.yaml
spec:
  image: cr.nebius.cloud/endoseg/preprocess:latest
  resources: { cpu: 4, memoryGiB: 8, gpu: 0 }
  command: ["python", "preprocess.py"]
  mounts:
    - objectStorageBucket: endoseg-data
```

Endpoint excerpt:

```yaml
# nebius/endpoint.yaml
spec:
  image: cr.nebius.cloud/endoseg/endpoint:latest
  resources: { gpu: 1, gpuType: nvidia-l40s }
  autoscaling: { minReplicas: 0, maxReplicas: 2 }
```

<!-- TODO: Insert screenshot of Job logs, GPU utilization, and public Endpoint URL after proof-of-execution run. -->

Object storage buckets (`endoseg-data`, `endoseg-checkpoints`) hold raw data, processed splits, and exported ONNX weights. The export Job writes `browser/model/unet.onnx` for the static demo.

## Browser Inference & Privacy by Design

The static demo under `browser/` loads the quantized ONNX model via a **Web Worker**. On first load the model bytes are cached in **IndexedDB** (`endoseg-cache` / `unet-v1`) so repeat visits skip the network fetch. Inference tries **WebGPU** first, falls back to **WASM**, applies sigmoid to logits, and renders a soft teal mask overlay with adjustable opacity and threshold.

The only optional network call is **Compare to cloud**, which POSTs a base64 PNG to a deploy-time proxy URL (`<meta name="proxy-url">`). The proxy forwards to the Nebius Endpoint with the secret token — the browser never sees API credentials.

Keyboard shortcut: press **Space** to segment after loading an image.

## Results

Trained on real MMOTU data (1469 images), 50 epochs, GPU (~25 min on an RTX 3050 Ti
laptop GPU). We report honestly, including where it falls short.

| Model | Split | Dice | IoU | Latency |
|---|---|---|---|---|
| U-Net (ONNX, fp32) | val | 0.7643 (epoch 33) | 0.6703 | _TODO_ |
| U-Net (ONNX, fp32) | test | 0.7545 | 0.6553 | _TODO_ |
| Foundation fine-tune | Nebius Endpoint (L40S) | _TODO_ | _TODO_ | _TODO_ |

<!-- TODO: Local vs cloud Dice delta, agreement %, browser latency on mid-range laptop. -->

**Honest limitation:** a visual sanity check on 8 random test-split images
(`jobs/finetune/visualize_predictions.py`) found per-sample Dice ranging from
**0.02 to 0.97** — the model is clearly not degenerate (it correctly localizes
medium/large lesions on 6 of 8 samples), but on small or thin lesions it tends to
confidently predict a larger, "typical"-shaped blob that doesn't match the true
lesion boundary. We think this comes from small lesions being underrepresented in
the training split, and we're calling it out rather than only reporting the
aggregate Dice, which would hide it.

## Try It Yourself

Serve the demo locally:

```bash
npx serve browser
# open http://localhost:3000
```

Upload an ultrasound frame or click **Try this** on a sample image, then press **Segment**. See `browser/deploy.md` for GitHub Pages and Netlify instructions.

Live demo: _TODO — link after GitHub Pages deploy_

## Reproducing This Project

```bash
git clone https://github.com/Gnana-Shishir-Kumar/nebius-challenge.git
cd nebius-challenge
pip install -r requirements.txt

# Verify MMOTU license, then download data
MMOTU_ARCHIVE_URL=<url> bash data/download_mmotu.sh

# Full pipeline (uncomment steps in repro.sh as implemented)
bash repro.sh
```

Build and push Docker images, then:

```bash
nebius serverless job create -f nebius/job_preprocess.yaml
nebius serverless job create -f nebius/job_finetune.yaml
nebius serverless endpoint create -f nebius/endpoint.yaml
```

Place the exported `unet.onnx` in `browser/model/` before running the demo.

## References

- MMOTU dataset — cite full bibliographic entry after license confirmation
- Nebius Serverless AI documentation — Jobs & Endpoints
- ONNX Runtime Web — WebGPU execution provider
- World Health Organization endometriosis fact sheet (190M global prevalence estimate)

---

#NebiusServerlessChallenge #MedicalAI #WebAssembly

Repository: https://github.com/Gnana-Shishir-Kumar/nebius-challenge
