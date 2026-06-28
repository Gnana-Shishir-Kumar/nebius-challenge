// EndoSeg browser app — UI glue. Heavy ONNX inference runs in worker.js so the
// UI thread stays responsive (PRD B8). This module handles image loading,
// drawing the overlay, and messaging the worker.

const IMG_SIZE = 256;
const MODEL_URL = "./models/unet_int8.onnx"; // produced by the export job
const PROXY_URL = "/infer"; // token-hiding proxy (set per deployment)

const els = {
  file: document.getElementById("fileInput"),
  gallery: document.getElementById("gallery"),
  run: document.getElementById("runBtn"),
  cloud: document.getElementById("cloudBtn"),
  opacity: document.getElementById("opacity"),
  backend: document.getElementById("backend"),
  latency: document.getElementById("latency"),
  delta: document.getElementById("delta"),
  canvas: document.getElementById("view"),
};
const ctx = els.canvas.getContext("2d");

// Sample gallery — replace with real anonymized MMOTU thumbnails in /samples.
const SAMPLES = ["samples/sample1.png", "samples/sample2.png", "samples/sample3.png"];

let currentImage = null; // ImageBitmap
let lastMask = null; // Uint8Array (256*256), local prediction

const worker = new Worker("./worker.js", { type: "module" });
worker.postMessage({ type: "init", modelUrl: MODEL_URL });

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "ready") {
    els.backend.textContent = msg.backend;
    if (currentImage) els.run.disabled = false;
  } else if (msg.type === "result") {
    lastMask = msg.mask;
    els.latency.textContent = `${msg.latencyMs.toFixed(0)} ms`;
    draw();
    els.cloud.disabled = false;
  } else if (msg.type === "error") {
    console.error("worker error:", msg.message);
    els.backend.textContent = "error";
  }
};

function buildGallery() {
  SAMPLES.forEach((src) => {
    const img = document.createElement("img");
    img.src = src;
    img.alt = "sample ultrasound";
    img.addEventListener("click", async () => {
      document.querySelectorAll(".gallery img").forEach((n) => n.classList.remove("active"));
      img.classList.add("active");
      await loadImage(src);
    });
    els.gallery.appendChild(img);
  });
}

async function loadImage(src) {
  const blob = typeof src === "string" ? await (await fetch(src)).blob() : src;
  currentImage = await createImageBitmap(blob);
  lastMask = null;
  draw();
  els.run.disabled = false;
  els.cloud.disabled = true;
}

function draw() {
  ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  if (currentImage) ctx.drawImage(currentImage, 0, 0, IMG_SIZE, IMG_SIZE);
  if (lastMask) overlayMask(lastMask, Number(els.opacity.value) / 100);
}

function overlayMask(mask, alpha) {
  const overlay = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      const p = i * 4;
      overlay.data[p] = Math.round(overlay.data[p] * (1 - alpha) + 255 * alpha); // R
      overlay.data[p + 1] = Math.round(overlay.data[p + 1] * (1 - alpha) + 77 * alpha); // G
      overlay.data[p + 2] = Math.round(overlay.data[p + 2] * (1 - alpha) + 109 * alpha); // B
    }
  }
  ctx.putImageData(overlay, 0, 0);
}

function getInputTensor() {
  // Resize via an offscreen canvas, then normalize to match training ([-1,1]).
  const off = new OffscreenCanvas(IMG_SIZE, IMG_SIZE);
  const octx = off.getContext("2d");
  octx.drawImage(currentImage, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = octx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const chw = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    chw[i] = (data[i * 4] / 255 - 0.5) / 0.5;
    chw[plane + i] = (data[i * 4 + 1] / 255 - 0.5) / 0.5;
    chw[2 * plane + i] = (data[i * 4 + 2] / 255 - 0.5) / 0.5;
  }
  return chw;
}

els.file.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) loadImage(f);
});

els.run.addEventListener("click", () => {
  if (!currentImage) return;
  els.run.disabled = true;
  const input = getInputTensor();
  worker.postMessage({ type: "infer", input, size: IMG_SIZE }, [input.buffer]);
  els.run.disabled = false;
});

els.opacity.addEventListener("input", draw);

els.cloud.addEventListener("click", async () => {
  // "Compare to cloud" — sends the image through the token-hiding proxy to the
  // Nebius Endpoint and reports the Dice delta vs the local mask.
  if (!currentImage || !lastMask) return;
  try {
    const blob = await currentImage // re-encode current image as PNG
      .then?.(() => null) || (await canvasToBlob());
    const fd = new FormData();
    fd.append("file", blob, "scan.png");
    const resp = await fetch(PROXY_URL, { method: "POST", body: fd });
    const json = await resp.json();
    const cloudMask = await decodeMaskPng(json.mask_png_base64);
    els.delta.textContent = dice(lastMask, cloudMask).toFixed(3);
  } catch (err) {
    console.error("cloud compare failed:", err);
    els.delta.textContent = "n/a";
  }
});

function canvasToBlob() {
  return new Promise((res) => els.canvas.toBlob(res, "image/png"));
}

async function decodeMaskPng(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const bmp = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
  const off = new OffscreenCanvas(IMG_SIZE, IMG_SIZE);
  const octx = off.getContext("2d");
  octx.drawImage(bmp, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = octx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const mask = new Uint8Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < mask.length; i++) mask[i] = data[i * 4] > 127 ? 1 : 0;
  return mask;
}

function dice(a, b) {
  let inter = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) {
    inter += a[i] & b[i];
    sa += a[i];
    sb += b[i];
  }
  return sa + sb === 0 ? 1 : (2 * inter) / (sa + sb);
}

buildGallery();
