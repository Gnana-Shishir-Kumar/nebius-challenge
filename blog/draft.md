# Fine-Tuned in the Cloud, Inferred in Your Browser: EndoSeg on Nebius Serverless

*Draft — target 600+ words. Tag `#NebiusServerlessChallenge` and link the repo on publish.*

> Research/education only. EndoSeg is not a medical device and makes no diagnostic claims.

## TL;DR

EndoSeg fine-tunes an ovarian-lesion ultrasound segmentation model on **Nebius
Serverless** (Jobs for preprocessing, fine-tuning, and ONNX export; an Endpoint
for full-precision cloud inference) and then runs a compact U-Net **fully
client-side in the browser** via ONNX Runtime Web + WebGPU. Your scan never
leaves your device.

## The problem

<!-- Endometriosis is under-served by ML tooling; public imaging data is
fragmented; almost no one ships an accessible, privacy-preserving demo. Frame
honestly: ovarian-lesion ultrasound segmentation that flags endometriomas — not
endometriosis diagnosis. -->

## Why this architecture

<!-- Two-model split:
- Browser model = compact 2D U-Net (ONNX-clean, quantized, <=50 MB, 256x256).
- Endpoint model = heavier foundation-model fine-tune for the "compare to cloud" feature.
Explain the post-processing caveat: SAM/Cellpose ops don't all map to ONNX/WebGPU,
so the browser ships the U-Net + JS connected-components. -->

## The Nebius pipeline

<!-- Walk through each Job + the Endpoint with a screenshot of logs/metrics:
1. Preprocess Job (N1): resize -> normalize -> split -> object storage.
2. Fine-tune Job (N2, GPU): train U-Net, log Dice/IoU per epoch, checkpoint.
3. Export Job (N3): ONNX export + quantize + parity check.
4. Endpoint (N4): full-precision HTTP inference, scale-to-zero.
Include the public Endpoint URL and a sample request/response. -->

## Privacy by design

<!-- Client-side inference; the only network call is the optional "compare to
cloud" through a thin token-hiding proxy that keeps the Nebius token server-side. -->

## Results (report honestly)

<!-- Dice / IoU on a patient-disjoint test split. Local (quantized) vs cloud
(full-precision) parity delta. Browser latency on a mid-range laptop (WebGPU). -->

| Model | Backend | Dice | IoU | Latency |
|---|---|---|---|---|
| U-Net (quantized) | Browser / WebGPU | _TBD_ | _TBD_ | _TBD_ |
| Foundation fine-tune | Nebius Endpoint | _TBD_ | _TBD_ | _TBD_ |

## Reproduce it

<!-- One-command: `bash repro.sh`. Link the repo, README, and dataset license notes. -->

## What I'd do next

<!-- 8-class semantic segmentation, GLENDA laparoscopy track, promptable SAM2,
human-in-the-loop brush. -->

## Citations

<!-- MMOTU (+ GLENDA if used) full citations. Nebius docs. -->
