// EndoSeg inference Web Worker.
// Loads the quantized ONNX model and runs inference off the UI thread.
// Tries WebGPU first, falls back to WASM (PRD B4). ORT is loaded via importScripts
// because module workers don't always resolve the CDN UMD bundle cleanly.

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js");

let session = null;
let backend = "wasm";

async function init(modelUrl) {
  // Prefer WebGPU; gracefully degrade to WASM.
  const tryOrder = ["webgpu", "wasm"];
  for (const ep of tryOrder) {
    try {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: [ep] });
      backend = ep;
      break;
    } catch (err) {
      console.warn(`provider ${ep} failed:`, err);
    }
  }
  if (!session) throw new Error("could not create ONNX session on any backend");
  postMessage({ type: "ready", backend });
}

async function infer(input, size) {
  const start = performance.now();
  const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
  const output = await session.run({ input: tensor });
  const logits = output[Object.keys(output)[0]].data; // Float32Array, size*size

  // sigmoid -> threshold 0.5 -> uint8 mask
  const mask = new Uint8Array(size * size);
  for (let i = 0; i < mask.length; i++) {
    const p = 1 / (1 + Math.exp(-logits[i]));
    mask[i] = p > 0.5 ? 1 : 0;
  }
  const latencyMs = performance.now() - start;
  postMessage({ type: "result", mask, latencyMs }, [mask.buffer]);
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") await init(msg.modelUrl);
    else if (msg.type === "infer") await infer(msg.input, msg.size);
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  }
};
