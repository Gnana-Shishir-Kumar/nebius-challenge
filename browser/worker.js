// EndoSeg inference Web Worker — ONNX Runtime Web (WebGPU → WASM fallback).

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let session = null;

async function loadModel(modelUrl) {
  const providers = ["webgpu", "wasm"];

  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [ep],
      });
      postMessage({ type: "modelLoaded", backend: ep });
      return;
    } catch (err) {
      console.warn(`execution provider ${ep} failed:`, err);
    }
  }

  throw new Error("could not create ONNX session on WebGPU or WASM");
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
