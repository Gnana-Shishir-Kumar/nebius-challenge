// EndoSeg browser app — UI + ONNX inference worker.

const MODEL_URL = "./model/unet.onnx";
const INFER_SIZE = 256;
const CANVAS_SIZE = 512;
const PRIVACY_DISMISSED_KEY = "endoseg-privacy-dismissed";
const MASK_COLOR = { r: 13, g: 122, b: 107 }; // #0d7a6b
const HEATMAP_OVERLAY_ALPHA = 0.6;

// --- State ---
const state = {
  model: null,
  backend: null,
  inferring: false,
  hasImage: false,
  currentImageElement: null,
  lastMaskData: null,
  letterbox: null,
  showHeatmap: false,
};

// --- DOM refs ---
const els = {
  privacyBanner: document.getElementById("privacy-banner"),
  privacyDismiss: document.getElementById("privacy-dismiss"),
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  outputCanvas: document.getElementById("output-canvas"),
  opacitySlider: document.getElementById("opacity-slider"),
  opacityValue: document.getElementById("opacity-value"),
  thresholdSlider: document.getElementById("threshold-slider"),
  thresholdValue: document.getElementById("threshold-value"),
  heatmapToggle: document.getElementById("heatmap-toggle"),
  segmentBtn: document.getElementById("segment-btn"),
  statusBar: document.getElementById("status-bar"),
  backendInfo: document.getElementById("backend-info"),
  latencyInfo: document.getElementById("latency-info"),
  compareCloudBtn: document.getElementById("compare-cloud-btn"),
  sampleTryBtns: document.querySelectorAll(".sample-card__try"),
  sampleThumbs: document.querySelectorAll(".sample-card__thumb"),
  confidenceReadout: document.getElementById("confidence-readout"),
  confMean: document.getElementById("conf-mean"),
  confMax: document.getElementById("conf-max"),
  confArea: document.getElementById("conf-area"),
};

const ctx = els.outputCanvas.getContext("2d");

// --- Worker ---
const worker = new Worker("./worker.js");

worker.onmessage = (e) => {
  const msg = e.data;

  if (msg.type === "modelLoaded") {
    state.model = true;
    state.backend = msg.backend;
    els.backendInfo.textContent = formatBackend(msg.backend);
    setStatus("Model loaded — upload an image to segment", "done");
  } else if (msg.type === "result") {
    onInferResult(msg.maskData, msg.latencyMs);
  } else if (msg.type === "error") {
    onInferError(msg.message);
  }
};

// --- Status helpers ---

function setStatus(text, status = "ready") {
  els.statusBar.textContent = text;
  els.statusBar.dataset.status = status;
}

function formatBackend(backend) {
  if (!backend) return "—";
  return backend === "webgpu" ? "WebGPU" : "WASM";
}

// --- Model loading ---

function loadModel() {
  setStatus("Loading model...", "running");
  els.backendInfo.textContent = "—";
  worker.postMessage({ type: "loadModel", modelUrl: MODEL_URL });
}

// --- Canvas helpers ---

function clearCanvas(fill = getComputedStyle(document.documentElement).getPropertyValue("--color-placeholder").trim()) {
  ctx.fillStyle = fill || "#c5cad0";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

function drawImageLetterboxed(source) {
  clearCanvas("#1a1d21");
  const sw = source.width;
  const sh = source.height;
  const scale = Math.min(CANVAS_SIZE / sw, CANVAS_SIZE / sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (CANVAS_SIZE - dw) / 2;
  const dy = (CANVAS_SIZE - dh) / 2;
  ctx.drawImage(source, dx, dy, dw, dh);
  state.letterbox = { dx, dy, dw, dh };
}

function drawPlaceholderSample(sampleId) {
  const off = document.createElement("canvas");
  off.width = CANVAS_SIZE;
  off.height = CANVAS_SIZE;
  const offCtx = off.getContext("2d");

  offCtx.fillStyle = "#c5cad0";
  offCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  offCtx.fillStyle = "#a8adb4";
  offCtx.fillRect(CANVAS_SIZE * 0.15, CANVAS_SIZE * 0.2, CANVAS_SIZE * 0.7, CANVAS_SIZE * 0.6);
  offCtx.fillStyle = "#8e949c";
  offCtx.beginPath();
  offCtx.ellipse(CANVAS_SIZE * 0.5, CANVAS_SIZE * 0.5, CANVAS_SIZE * 0.22, CANVAS_SIZE * 0.18, 0, 0, Math.PI * 2);
  offCtx.fill();
  offCtx.fillStyle = "#6b7179";
  offCtx.font = "600 14px system-ui, sans-serif";
  offCtx.textAlign = "center";
  offCtx.fillText(`Sample ${sampleId} placeholder`, CANVAS_SIZE / 2, CANVAS_SIZE * 0.88);

  state.currentImageElement = off;
  state.lastMaskData = null;
  hideConfidenceReadout();
  drawImageLetterboxed(off);
}

// --- Rendering utilities ---

function getMaskOpacity() {
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--mask-opacity").trim();
  const parsed = parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : 0.7;
}

function getThreshold() {
  return Number(els.thresholdSlider.value);
}

function hslToRgb(h, s, l) {
  const sat = s / 100;
  const light = l / 100;
  const c = (1 - Math.abs(2 * light - 1)) * sat;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = light - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;

  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];

  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

function drawMaskLegend() {
  const padding = 10;
  const swatch = 12;
  const labelY = CANVAS_SIZE - padding - 6;

  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.fillRect(padding - 4, CANVAS_SIZE - padding - swatch - 14, 78, 24);

  ctx.fillStyle = "#0d7a6b";
  ctx.fillRect(padding, CANVAS_SIZE - padding - swatch - 8, swatch, swatch);

  ctx.fillStyle = "#ffffff";
  ctx.font = "600 11px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText("Lesion", padding + swatch + 6, labelY);
  ctx.restore();
}

function renderMask(maskData, threshold = getThreshold()) {
  if (!state.currentImageElement || !maskData) return;

  // Step 1: letterboxed original image at display size.
  drawImageLetterboxed(state.currentImageElement);

  const maskOpacity = getMaskOpacity();

  // Step 2: 256×256 mask ImageData with soft probability-weighted alpha.
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = INFER_SIZE;
  maskCanvas.height = INFER_SIZE;
  const maskCtx = maskCanvas.getContext("2d");
  const imageData = maskCtx.createImageData(INFER_SIZE, INFER_SIZE);

  for (let i = 0; i < maskData.length; i++) {
    const prob = maskData[i];
    if (prob <= threshold) continue;

    const p = i * 4;
    imageData.data[p] = MASK_COLOR.r;
    imageData.data[p + 1] = MASK_COLOR.g;
    imageData.data[p + 2] = MASK_COLOR.b;
    imageData.data[p + 3] = Math.round(maskOpacity * prob * 255);
  }

  maskCtx.putImageData(imageData, 0, 0);

  // Step 3: scale mask overlay to full canvas.
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(maskCanvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.restore();

  // Step 5: legend.
  drawMaskLegend();

  // Step 4: persist for slider redraws.
  state.lastMaskData = maskData;
}

function renderHeatmap(maskData) {
  if (!state.currentImageElement || !maskData) return;

  drawImageLetterboxed(state.currentImageElement);

  const heatCanvas = document.createElement("canvas");
  heatCanvas.width = INFER_SIZE;
  heatCanvas.height = INFER_SIZE;
  const heatCtx = heatCanvas.getContext("2d");
  const imageData = heatCtx.createImageData(INFER_SIZE, INFER_SIZE);

  for (let i = 0; i < maskData.length; i++) {
    const prob = maskData[i];
    if (prob <= 0) continue;

    const hue = 240 - prob * 240;
    const [r, g, b] = hslToRgb(hue, 80, 50);
    const p = i * 4;
    imageData.data[p] = r;
    imageData.data[p + 1] = g;
    imageData.data[p + 2] = b;
    imageData.data[p + 3] = Math.round(prob * 255);
  }

  heatCtx.putImageData(imageData, 0, 0);

  ctx.save();
  ctx.globalAlpha = HEATMAP_OVERLAY_ALPHA;
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(heatCanvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.restore();

  state.lastMaskData = maskData;
}

function computeConfidenceStats(maskData, threshold = getThreshold()) {
  let maxProb = 0;
  let lesionCount = 0;
  let lesionSum = 0;

  for (let i = 0; i < maskData.length; i++) {
    const prob = maskData[i];
    if (prob > maxProb) maxProb = prob;
    if (prob > threshold) {
      lesionCount += 1;
      lesionSum += prob;
    }
  }

  const total = maskData.length;
  const meanInLesion = lesionCount > 0 ? lesionSum / lesionCount : 0;
  const areaPct = (lesionCount / total) * 100;

  return {
    meanInLesion,
    maxProb,
    areaPct,
  };
}

function updateConfidenceReadout(maskData, threshold = getThreshold()) {
  if (!maskData) {
    hideConfidenceReadout();
    return;
  }

  const { meanInLesion, maxProb, areaPct } = computeConfidenceStats(maskData, threshold);

  els.confMean.textContent = `${(meanInLesion * 100).toFixed(1)}%`;
  els.confMax.textContent = `${(maxProb * 100).toFixed(1)}%`;
  els.confArea.textContent = `${areaPct.toFixed(1)}%`;
  els.confidenceReadout.hidden = false;
}

function hideConfidenceReadout() {
  els.confidenceReadout.hidden = true;
  els.confMean.textContent = "—";
  els.confMax.textContent = "—";
  els.confArea.textContent = "—";
}

function redrawMaskOverlay() {
  if (!state.lastMaskData) return;

  const threshold = getThreshold();

  if (state.showHeatmap) {
    renderHeatmap(state.lastMaskData);
  } else {
    renderMask(state.lastMaskData, threshold);
  }

  updateConfidenceReadout(state.lastMaskData, threshold);
}

// --- Inference ---

function imageToGrayscaleTensor(imageElement) {
  const off = document.createElement("canvas");
  off.width = INFER_SIZE;
  off.height = INFER_SIZE;
  const offCtx = off.getContext("2d");
  offCtx.drawImage(imageElement, 0, 0, INFER_SIZE, INFER_SIZE);

  const { data } = offCtx.getImageData(0, 0, INFER_SIZE, INFER_SIZE);
  const imageData = new Float32Array(INFER_SIZE * INFER_SIZE);

  for (let i = 0; i < INFER_SIZE * INFER_SIZE; i++) {
    const p = i * 4;
    imageData[i] = (data[p] * 0.299 + data[p + 1] * 0.587 + data[p + 2] * 0.114) / 255;
  }

  return imageData;
}

async function runInference(imageElement) {
  if (!imageElement || !state.model || state.inferring) return;

  state.inferring = true;
  els.segmentBtn.disabled = true;
  setStatus("Running inference...", "running");

  const imageData = imageToGrayscaleTensor(imageElement);

  worker.postMessage(
    {
      type: "infer",
      imageData,
      width: INFER_SIZE,
      height: INFER_SIZE,
    },
    [imageData.buffer],
  );
}

function onInferResult(maskData, workerLatencyMs) {
  state.inferring = false;
  state.lastMaskData = maskData;

  redrawMaskOverlay();
  els.latencyInfo.textContent = `${Math.round(workerLatencyMs)} ms`;
  els.backendInfo.textContent = formatBackend(state.backend);
  els.segmentBtn.disabled = !state.hasImage;
  setStatus("Segmentation complete", "done");
}

function onInferError(message) {
  state.inferring = false;
  els.segmentBtn.disabled = state.hasImage;

  if (!state.model) {
    els.backendInfo.textContent = "—";
    setStatus(`Model load failed: ${message}`, "error");
    return;
  }

  setStatus(`Inference failed: ${message}`, "error");
}

// --- Privacy banner dismiss ---

function initPrivacyBanner() {
  if (sessionStorage.getItem(PRIVACY_DISMISSED_KEY) === "1") {
    els.privacyBanner.classList.add("is-hidden");
  }

  els.privacyDismiss.addEventListener("click", () => {
    els.privacyBanner.classList.add("is-hidden");
    sessionStorage.setItem(PRIVACY_DISMISSED_KEY, "1");
  });
}

// --- Upload & drag-drop ---

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose a valid image file", "error");
    return;
  }

  const url = URL.createObjectURL(file);
  const img = new Image();

  img.onload = () => {
    state.currentImageElement = img;
    state.lastMaskData = null;
    hideConfidenceReadout();
    drawImageLetterboxed(img);
    URL.revokeObjectURL(url);
    state.hasImage = true;
    els.segmentBtn.disabled = state.inferring;
    setStatus("Image loaded — click Segment", "done");
  };

  img.onerror = () => {
    URL.revokeObjectURL(url);
    setStatus("Could not load image", "error");
  };

  img.src = url;
}

function initUpload() {
  els.dropZone.addEventListener("click", () => els.fileInput.click());

  els.dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });

  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  });

  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("is-dragover");
  });

  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("is-dragover");
  });

  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("is-dragover");
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });
}

// --- Sample gallery ---

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${url}`));
    img.src = url;
  });
}

async function loadSample(sampleId, src) {
  try {
    const img = await loadImageFromUrl(src);
    state.currentImageElement = img;
    state.lastMaskData = null;
    hideConfidenceReadout();
    drawImageLetterboxed(img);
    state.hasImage = true;
    els.segmentBtn.disabled = state.inferring;
    setStatus("Image loaded — click Segment", "done");
  } catch {
    drawPlaceholderSample(sampleId);
    state.hasImage = true;
    els.segmentBtn.disabled = state.inferring;
    setStatus("Image loaded — click Segment", "done");
  }
}

function initSampleGallery() {
  els.sampleThumbs.forEach((thumb) => {
    thumb.addEventListener("error", () => {
      thumb.removeAttribute("src");
      thumb.classList.add("sample-card__thumb--placeholder");
    });
  });

  els.sampleTryBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const sampleId = btn.dataset.sample;
      const src = btn.dataset.sampleSrc;
      loadSample(sampleId, src);
    });
  });
}

// --- Sliders & toggles ---

function initOpacitySlider() {
  const updateOpacity = () => {
    const pct = Number(els.opacitySlider.value);
    document.documentElement.style.setProperty("--mask-opacity", String(pct / 100));
    els.opacityValue.textContent = `${pct}%`;
    redrawMaskOverlay();
  };

  els.opacitySlider.addEventListener("input", updateOpacity);
  updateOpacity();
}

function initThresholdSlider() {
  const updateThreshold = () => {
    const value = Number(els.thresholdSlider.value);
    els.thresholdValue.textContent = value.toFixed(2);
    redrawMaskOverlay();
  };

  els.thresholdSlider.addEventListener("input", updateThreshold);
  updateThreshold();
}

function initHeatmapToggle() {
  els.heatmapToggle.addEventListener("change", () => {
    state.showHeatmap = els.heatmapToggle.checked;
    redrawMaskOverlay();
  });
}

// --- Segment button ---

function initSegmentButton() {
  els.segmentBtn.addEventListener("click", () => {
    if (!state.hasImage || !state.currentImageElement || state.inferring) return;
    runInference(state.currentImageElement);
  });
}

// --- init() ---

function init() {
  clearCanvas();
  initPrivacyBanner();
  initUpload();
  initSampleGallery();
  initOpacitySlider();
  initThresholdSlider();
  initHeatmapToggle();
  initSegmentButton();
  loadModel();
}

document.addEventListener("DOMContentLoaded", init);

// DevTools helpers for manual mask testing (see verification steps).
if (typeof window !== "undefined") {
  window.__endoseg = { renderMask, renderHeatmap, redrawMaskOverlay, state };
}
