# Product Requirements Document — "EndoSeg"
### Browser-Native Gynecological Image Segmentation, Fine-Tuned on Nebius Serverless

| Field | Value |
|---|---|
| **Working title** | EndoSeg (rename freely) |
| **Tagline** | *"Fine-tuned in the cloud. Inferred privately in your browser — your scan never leaves your laptop."* |
| **Challenge** | Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences track, "Medical image segmentation endpoints" |
| **Owner** | _[your name]_ |
| **Team** | Up to 3 members |
| **Version** | Draft v0.1 — for review |
| **Status** | Pre-build / scoping |
| **Target submission** | Confirm deadline on official page (reported July 15, 2026; announcement said June 30 — **verify first**) |

---

## 1. Problem & Opportunity

Endometriosis affects ~10% of reproductive-age women (≈190M worldwide) yet is under-served by ML tooling, and is diagnosed late on average. Public, anonymized imaging data exists but is fragmented, and almost no one ships an accessible, privacy-preserving demo that a non-specialist can actually try.

**The opportunity for this challenge:** combine three things that rarely appear together in a single submission —
1. A real medical segmentation fine-tuning pipeline on **Nebius Serverless** (deep product usage),
2. **Client-side, in-browser inference** so medical images never leave the user's device (privacy by design, genuinely original), and
3. An **honest architecture comparison** (U-Net vs MedSAM/SAM2 vs nnU-Net vs Cellpose-SAM) packaged as an educational tutorial.

This directly maps to all six judging criteria: technical implementation, reproducibility, educational content quality, product usage depth, real-world usefulness, originality.

---

## 2. Goals & Non-Goals

### 2.1 Goals
- **G1.** Place in or win the Healthcare & Life Sciences track (top prize $2,000 Nebius credits; every valid submission earns $100).
- **G2.** Ship a working browser demo that segments gynecological ultrasound (and optionally MRI) images **fully client-side**.
- **G3.** Demonstrate **deep Nebius usage**: both Serverless **Jobs** (preprocessing, fine-tuning, batch eval/export) and a Serverless **Endpoint** (cloud inference / comparison / fallback).
- **G4.** Be **fully reproducible** by another practitioner from the README alone, using only public/anonymized data.
- **G5.** Publish a high-quality 600+ word technical blog (tagged `#NebiusServerlessChallenge`) and a 3–10 min video.

### 2.2 Non-Goals
- **NG1.** Not a medical device. No diagnostic claims; research/education only.
- **NG2.** Not using any patient/private data — public, anonymized, or synthetic only.
- **NG3.** Not solving 3D MRI in the browser (slice to 2D or keep on the Endpoint).
- **NG4.** Not aiming for clinical-grade Dice on hard targets (deep infiltrating endometriosis is genuinely hard; ovary contours have poor inter-rater agreement).

---

## 3. Target "Users"

| Persona | What they need | How EndoSeg serves them |
|---|---|---|
| **Challenge judge** (Nebius engineer / domain expert) | Reproduce results fast, see real Nebius usage, learn something | One-command repro, committed Job/Endpoint configs, comparison writeup |
| **ML practitioner** (blog reader) | A pattern they can adapt: cloud fine-tune → browser inference | Tutorial blog + clean repo |
| **Demo visitor** (clinician-curious, journalist, recruiter) | Try it instantly, understand the privacy angle | Static webpage, sample gallery, "image never leaves device" banner |

---

## 4. Success Metrics

### 4.1 Challenge outcome (primary)
- Valid submission accepted (repo + blog + Nebius proof-of-execution).
- Strong score on product usage depth (both Jobs + Endpoint used and documented).
- Visible community signal (blog engagement; helps tie-breaks).

### 4.2 Technical (secondary, reported honestly)
- **Dice / IoU per class** on a held-out, patient-disjoint test split (report real numbers, don't fabricate).
- **Browser inference latency** target ≤ ~2–3 s on a mid-range laptop (256×256 input, WebGPU).
- **Model size** ≤ ~50 MB (quantized) for the browser model.
- **Local vs cloud parity**: quantified accuracy delta between browser (quantized) and Endpoint (full-precision) model.

---

## 5. Scope: MVP vs Stretch

**Locked decision:** 2D ultrasound only. **No MRI, no 3D→2D slicing** anywhere in the MVP (avoids the heaviest engineering and the largest browser risk).

### 5.1 MVP (must ship — this alone is a complete, competitive entry)
- Dataset: **MMOTU** ovarian-tumor ultrasound (2D, pixel masks, 8 tumor classes including endometrioma / "chocolate cyst").
- **Segmentation task (MVP default): binary** — ovarian lesion vs background — plus a lightweight **type label** that flags when the lesion is an endometrioma. (Full 8-class semantic segmentation is a stretch; see §6.1.)
- **Two-model split** (resolves the earlier browser-export risk):
  - **Browser model = compact 2D U-Net** fine-tuned on MMOTU → exports cleanly to ONNX, runs client-side.
  - **Endpoint model = foundation-model fine-tune** (UltraSam / MedSAM2 / SAM2) → heavier, higher accuracy, lives on the Nebius Endpoint for the "compare to cloud" feature.
- **Nebius Job** for preprocessing + fine-tuning (both models) + ONNX export.
- **Nebius Endpoint** hosting the foundation-model fine-tune for cloud comparison/fallback.
- **Browser app**: upload/sample image → client-side ONNX inference → mask overlay.
- Repo + Dockerfile + README + 600-word blog.

### 5.2 Stretch (each adds standout value — all 2D, no slicing)
- **Full 8-class semantic segmentation** on MMOTU (instead of binary), with per-class Dice.
- **MedSAM / SAM2 promptable model** running in-browser (click-to-segment) — validate ONNX export early.
- **Cellpose-style human-in-the-loop** correction tool in the browser (brush to fix mask → optionally feed back).
- **GLENDA laparoscopy track** (2D endometriosis-lesion frames — *no slicing needed*; this is now the primary "true endometriosis" extension since the MRI track is dropped).
- **Architecture bake-off** figure: U-Net vs MedSAM vs UltraSam vs Cellpose-SAM with real metrics.

---

## 6. Feature List (prioritized)

Priority key: **P0** = MVP/blocking, **P1** = high-value stretch, **P2** = nice-to-have.

### 6.1 Data pipeline
| ID | Feature | Priority |
|---|---|---|
| D1 | Download + verify MMOTU (license check, integrity) | P0 |
| D2 | Preprocessing: resize→256×256, normalize (CLAHE/intensity), RGB-convert, mask encode | P0 |
| D3 | Patient-disjoint train/val/test split (no leakage) | P0 |
| D4 | Augmentation (Albumentations: flips, elastic, intensity, noise) | P0 |
| D5 | **Task scoping**: binary lesion mask (P0) → optional 8-class semantic masks (P1) | P0 |
| D6 | Class-balancing / oversampling for the minority endometrioma class | P1 |
| D7 | GLENDA laparoscopy frame ingest (COCO → mask) — 2D, no slicing | P2 |

> **Removed:** UT-EndoMRI / any 3D→2D MRI slicing. Out of scope per the ultrasound-only decision.

### 6.2 Model & training
| ID | Feature | Priority |
|---|---|---|
| M1 | **Compact 2D U-Net** (browser model) — binary lesion; ONNX-clean | P0 |
| M2 | Metrics: Dice, IoU, per-class, confusion | P0 |
| M3 | ONNX export + fp16/int8 quantization + weight sharding | P0 |
| M4 | **Foundation-model fine-tune** (UltraSam / MedSAM2 / SAM2) — Endpoint model | P0 |
| M5 | Endometrioma type-label head (flag chocolate cyst) | P1 |
| M6 | Cellpose-SAM human-in-the-loop fine-tuning track | P1 |
| M7 | Architecture comparison harness (same split, same metrics) | P1 |
| M8 | Model registry / checkpoint versioning in object storage | P2 |

### 6.3 Nebius Serverless integration
| ID | Feature | Priority |
|---|---|---|
| N1 | **Job**: preprocessing/ETL container | P0 |
| N2 | **Job**: fine-tuning container (GPU) | P0 |
| N3 | **Job**: batch evaluation + ONNX export | P0 |
| N4 | **Endpoint**: hosted full-precision inference (HTTP) | P0 |
| N5 | Object-storage mounts for data + checkpoints | P0 |
| N6 | Logs + GPU/utilization + cost capture (screenshots for blog) | P0 |
| N7 | Thin proxy to hide Nebius token from the browser | P1 |
| N8 | Endpoint serves the heavier foundation-model the browser can't run full-precision | P1 |

### 6.4 Browser app (client-side)
| ID | Feature | Priority |
|---|---|---|
| B1 | Static site, no backend needed for local inference | P0 |
| B2 | Drag-and-drop / file upload | P0 |
| B3 | **Sample image gallery** (users won't have their own scans) | P0 |
| B4 | ONNX Runtime Web inference, WebGPU + WASM fallback | P0 |
| B5 | Mask overlay with opacity slider + multi-class color legend | P0 |
| B6 | Backend indicator (WebGPU vs WASM) + latency readout | P0 |
| B7 | Privacy banner: "your image never leaves this device" | P0 |
| B8 | Inference in a Web Worker (keep UI responsive) | P1 |
| B9 | Model caching (IndexedDB / Service Worker) | P1 |
| B10 | **"Compare to cloud"** button → calls Nebius Endpoint, shows delta | P1 |
| B11 | Probability/confidence heatmap toggle | P1 |
| B12 | **Human-in-the-loop brush** to correct masks + download corrected mask | P1 |
| B13 | Model picker (U-Net vs MedSAM) | P2 |
| B14 | Click-to-segment (SAM2 promptable) | P2 |

### 6.5 Educational & submission
| ID | Feature | Priority |
|---|---|---|
| E1 | 600+ word blog post, tagged `#NebiusServerlessChallenge` | P0 |
| E2 | README: setup, hardware config, expected outputs, runtime/cost | P0 |
| E3 | Dockerfile(s) + open-source license (MIT/Apache-2.0) | P0 |
| E4 | One-command reproduction script | P0 |
| E5 | Architecture diagram (data → Nebius Jobs → Endpoint + browser) | P0 |
| E6 | 3–10 min video walkthrough | P1 |
| E7 | Proof-of-execution gallery (Job logs, Endpoint URL, metrics) | P0 |

---

## 7. System Architecture

```
            PUBLIC DATA (MMOTU / UT-EndoMRI / GLENDA)
                          │
                          ▼
            ┌─────────────────────────────┐
            │  Nebius Object Storage       │  ← raw + processed + checkpoints
            └─────────────────────────────┘
              │           │            ▲
              ▼           ▼            │
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ Job N1       │ │ Job N2       │ │ Job N3       │
   │ Preprocess   │→│ Fine-tune    │→│ Eval + ONNX  │
   │ /ETL         │ │ (GPU)        │ │ export+quant │
   └──────────────┘ └──────────────┘ └──────────────┘
                                         │        │
                  full-precision model   │        │  quantized ONNX
                          ▼              │        ▼
                  ┌──────────────┐       │   ┌──────────────────────┐
                  │ Endpoint N4  │◄──────┘   │  Static Browser App  │
                  │ cloud infer  │           │  ORT Web + WebGPU    │
                  └──────────────┘           │  (client-side)       │
                          ▲                  └──────────────────────┘
                          │  "compare to cloud" / 3D path  │
                          └────────────────────────────────┘
                              (via thin token-hiding proxy)
```

**Key design principle:** Nebius does all the *heavy and server-side* work (training + the big/3D model); the browser does *private, local* inference. The two meet only at the optional "compare to cloud" feature — which is itself a great demo of the Endpoint.

---

## 8. Technical Requirements

- **Languages/frameworks:** Python (PyTorch, MONAI/nnU-Net, Albumentations, ONNX) for training; JS/TS (ONNX Runtime Web, WebGPU) for the browser; static hosting (GitHub Pages / Netlify / Vercel).
- **Model targets:** browser model ≤ ~50 MB quantized, 256×256 input, batch size 1; full model on Endpoint unconstrained.
- **Export path:** PyTorch → ONNX → quantize (fp16/int8) → validate parity → shard if needed.
- **Post-processing caveat:** some foundation-model (SAM/MedSAM/Cellpose) post-processing ops don't all map cleanly to ONNX/WebGPU kernels — so the **browser** ships the compact semantic **U-Net + JS connected-components**; the heavier foundation-model fine-tune stays on the **Endpoint**.
- **Reproducibility:** pinned dependencies, fixed seeds, committed Job/Endpoint configs, dataset download scripts, Docker.
- **Compliance:** public/anonymized data only; **verify MMOTU's exact license before building on it** (confirm it permits research use + redistribution of derived models); cite all datasets; if GLENDA is used note it is CC BY-NC / research-only; display a research-only disclaimer.

---

## 9. Step-by-Step Implementation Flow

### Phase 0 — Setup & baseline (Days 1–2)
1. **Verify the live challenge deadline + exact rubric** on the official Nebius page (resolve June 30 vs July 15).
2. Create/confirm Nebius account; provision Serverless access; claim trial credits.
3. Create the public repo; add license (MIT/Apache-2.0), `.gitignore`, README skeleton, Dockerfile scaffold.
4. Download **MMOTU**; review license; sanity-check images + masks; visualize a few overlays.
5. Train a quick **local** 2D U-Net on a MMOTU subset to confirm the data + loss + metric loop works end-to-end before touching Nebius.

### Phase 1 — Nebius pipeline + fine-tuning (Days 3–6)
6. Containerize the **preprocessing Job (N1)**: resize/normalize/split → push processed data to object storage. Run on Nebius; capture logs.
7. Containerize the **fine-tuning Job (N2)**: (a) train the compact **browser U-Net**; (b) fine-tune the **foundation model** (UltraSam/MedSAM2/SAM2) for the Endpoint. Log Dice/IoU per epoch; checkpoint to object storage.
8. Containerize the **eval + export Job (N3)**: held-out evaluation for both models; export the U-Net to ONNX; quantize; validate ONNX parity vs PyTorch.
9. Stand up the **Endpoint (N4)** hosting the fine-tuned foundation model; test with a sample request; record the public URL.
10. **Lock the MVP models.** Capture proof-of-execution (Job logs, Endpoint response, metrics table) for the blog.

### Phase 2 — Browser app (Days 7–10)
11. Build the static app shell (B1): upload (B2), sample gallery (B3), privacy banner (B7).
12. Integrate **ONNX Runtime Web** (B4): load quantized model, WebGPU with WASM fallback, backend + latency indicator (B6).
13. Implement preprocessing-in-JS to match training (same resize/normalize) — *critical for parity*.
14. Render **mask overlay** (B5): opacity slider, color legend.
15. Move inference into a **Web Worker** (B8) + **model caching** (B9).
16. Wire the **"Compare to cloud"** button (B10) → Endpoint via the token-hiding proxy (N7); show local vs cloud latency + accuracy delta.
17. Reproducibility checkpoint: confirm a fresh clone + README steps reproduce training and the browser demo.

### Phase 3 — Depth, polish, content (Days 11–14)
18. **Stretch model track (all 2D, no slicing):** upgrade MMOTU to **full 8-class** semantic segmentation, and/or add the **GLENDA laparoscopy** track via a new Job served through the Endpoint (D7, N8).
19. Optional **MedSAM/SAM2** promptable model + **human-in-the-loop brush** (M4, B12, B14) and/or **Cellpose-SAM** HITL track (M6).
20. Build the **architecture comparison** figure + table (M7, E5): U-Net vs MedSAM vs UltraSam vs Cellpose-SAM, same split, honest numbers.
21. Write the **blog post** (E1): problem → data → architecture choices → Nebius pipeline → browser inference + privacy → results → repro. Tag `#NebiusServerlessChallenge`, link the repo.
22. Finalize **README** (E2): setup, hardware, expected outputs, runtime/cost, proof-of-execution gallery (E7).
23. Record the **video** (E6).

### Phase 4 — Reproducibility audit & submit (Days 15–17, buffer)
24. Clean-machine test: fresh clone → Docker → one-command repro (E4) → browser demo loads + infers.
25. Final license/compliance + disclaimer pass; remove any secrets.
26. Submit via the Submit tab before the deadline; include repo, blog, video, and proof-of-execution.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MMOTU "too small" for fine-tuning | Non-issue — fine-tune a foundation model (UltraSam/MedSAM2); 1.4k dense masks is ample |
| Endometrioma is only 1 of 8 MMOTU classes (minority) | Class-balancing/oversampling (D6); MVP is binary lesion seg + endometrioma flag, not per-type seg |
| ONNX post-processing gaps (SAM/Cellpose ops) | Browser = semantic U-Net + JS connected-components; foundation model stays on Endpoint |
| Browser latency/memory on weak laptops | Quantize, lower resolution, WASM fallback, or route to Endpoint |
| Overfitting | Heavy augmentation + transfer learning from a pretrained foundation model |
| **Overclaiming the "endometriosis" angle** | Frame honestly as ovarian-lesion ultrasound segmentation that detects endometriomas — not endometriosis diagnosis (see §2 non-goals) |
| MMOTU license terms | Verify before building; cite; research/education use only; no commercial claims |
| Nebius Serverless preview rough edges | Budget buffer days; keep a local training fallback |
| Deadline ambiguity | Verify on official page Day 1 |

---

## 11. Deliverables & Submission Checklist

- [ ] Public GitHub/GitLab repo: code using Nebius Jobs **and** Endpoints
- [ ] Dockerfile(s) + open-source license + no committed secrets/private data
- [ ] README: setup, hardware config, expected outputs, runtime/cost, repro steps
- [ ] Proof-of-execution: Job logs, Endpoint URL, metrics, screenshots
- [ ] Browser demo deployed (static host) + linked
- [ ] 600+ word technical blog, tagged `#NebiusServerlessChallenge`, links repo
- [ ] (Recommended) 3–10 min video walkthrough
- [ ] Research-only disclaimer + dataset citations
- [ ] Submitted via Submit tab before deadline

---

## 12. Open Decisions (need your input)

1. ✅ **Modality — RESOLVED:** 2D ultrasound (MMOTU), no MRI, no slicing.
2. **Confirm deadline & exact six-criteria weights** from the official page (still open — do Day 1).
3. **Segmentation task:** confirm MVP is **binary lesion seg + endometrioma flag** (recommended) vs jumping straight to 8-class. — your call.
4. **Stretch ambition:** which of {8-class seg, GLENDA laparoscopy, MedSAM/SAM2 promptable, Cellpose-SAM HITL} do you commit to? (MRI track is dropped.)
5. **Team split:** who owns (a) Nebius pipeline, (b) browser app, (c) content/blog/video?
6. **Hosting:** GitHub Pages vs Netlify vs Vercel for the static demo?
7. **Name:** keep "EndoSeg" or pick something more brandable?

---

*This PRD is a working draft for review. Numbers, scope, and timeline should be adjusted after you confirm the official challenge rules and decide the stretch ambition.*
