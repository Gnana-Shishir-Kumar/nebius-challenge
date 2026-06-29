// EndoSeg inference Web Worker — ONNX Runtime Web (WebGPU → WASM fallback).

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

const CACHE_DB = "endoseg-cache";
const CACHE_STORE = "models";
const CACHE_KEY = "unet-v1";

let session = null;

// --- IndexedDB helpers ---

function openCacheDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(CACHE_DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(CACHE_STORE)) {
        db.createObjectStore(CACHE_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getModelFromCache(key) {
  const db = await openCacheDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CACHE_STORE, "readonly");
    const req = tx.objectStore(CACHE_STORE).get(key);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

async function saveModelToCache(key, buffer) {
  const db = await openCacheDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CACHE_STORE, "readwrite");
    tx.objectStore(CACHE_STORE).put(buffer, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function createSessionFromBuffer(buffer) {
  const providers = ["webgpu", "wasm"];

  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(buffer, {
        executionProviders: [ep],
      });
      return ep;
    } catch (err) {
      console.warn(`execution provider ${ep} failed:`, err);
    }
  }

  throw new Error("could not create ONNX session on WebGPU or WASM");
}

async function loadModel(modelUrl) {
  let buffer = await getModelFromCache(CACHE_KEY);
  let fromCache = false;

  if (buffer) {
    fromCache = true;
    console.log("Loaded model from cache");
  } else {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`failed to fetch model: ${response.status} ${response.statusText}`);
    }
    buffer = await response.arrayBuffer();
    await saveModelToCache(CACHE_KEY, buffer);
  }

  const backend = await createSessionFromBuffer(buffer);
  postMessage({ type: "modelLoaded", backend, fromCache });
}

async function infer(imageData, width, height) {
  if (!session) {
    throw new Error("model not loaded");
  }

  const start = performance.now();
  const tensor = new ort.Tensor("float32", imageData, [1, 1, height, width]);
  const output = await session.run({ input: tensor });
  const logits = output[Object.keys(output)[0]].data;

  const maskData = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    maskData[i] = 1 / (1 + Math.exp(-logits[i]));
  }

  const latencyMs = performance.now() - start;
  postMessage({ type: "result", maskData, latencyMs }, [maskData.buffer]);
}

onmessage = async (e) => {
  const msg = e.data;

  try {
    if (msg.type === "loadModel") {
      await loadModel(msg.modelUrl);
    } else if (msg.type === "infer") {
      await infer(msg.imageData, msg.width, msg.height);
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err?.message || err) });
  }
};
