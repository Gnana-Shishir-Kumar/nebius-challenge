"""Nebius Endpoint N4 — hosted full-precision segmentation inference (HTTP).

Serves the heavier model (foundation-model fine-tune in the full build; the
U-Net here as a working default) behind a small FastAPI app. The browser's
"Compare to cloud" button hits this endpoint via the token-hiding proxy.

POST /infer  multipart image file  -> PNG mask (base64) + latency + meta
GET  /health                       -> liveness probe for the Endpoint
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # keeps the stub importable without ORT installed
    ort = None

IMG_SIZE = int(os.getenv("IMG_SIZE", "256"))
MODEL_PATH = os.getenv("MODEL_PATH", "/models/unet.onnx")

app = FastAPI(title="EndoSeg Endpoint", version="0.1.0")
_session: "ort.InferenceSession | None" = None


def get_session() -> "ort.InferenceSession":
    """Lazy-load the ONNX session so cold start is the only heavy hit."""
    global _session
    if _session is None:
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        _session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print(f"loaded model {MODEL_PATH} with providers {providers}")
    return _session


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the training transform: resize -> RGB -> normalize [-1, 1] -> NCHW."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr.astype(np.float32)


def postprocess(logits: np.ndarray) -> np.ndarray:
    """Sigmoid -> threshold -> uint8 mask {0,255}."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    mask = (probs[0, 0] > 0.5).astype(np.uint8) * 255
    return mask


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": Path(MODEL_PATH).name}


@app.post("/infer")
async def infer(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    image = Image.open(io.BytesIO(raw))
    start = time.perf_counter()
    inp = preprocess(image)
    logits = get_session().run(None, {"input": inp})[0]
    mask = postprocess(logits)
    latency_ms = (time.perf_counter() - start) * 1000.0

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse(
        {
            "mask_png_base64": mask_b64,
            "size": IMG_SIZE,
            "latency_ms": round(latency_ms, 2),
            "model": Path(MODEL_PATH).name,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
