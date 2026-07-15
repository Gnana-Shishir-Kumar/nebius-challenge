"""Nebius Endpoint N4 — hosted segmentation inference over HTTP (JSON)."""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from analysis import analyze_from_pil

MODEL_PATH = os.getenv("MODEL_PATH", "/model/unet.onnx")
IMG_SIZE = int(os.getenv("IMG_SIZE", "256"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "unet-v1")

app = FastAPI(title="EndoSeg Endpoint", version="0.1.0")
_session: "ort.InferenceSession | None" = None


class PredictRequest(BaseModel):
    image_b64: str  # base64-encoded PNG or JPG


class PredictResponse(BaseModel):
    mask_b64: str       # base64-encoded PNG mask (256x256 uint8, values 0-255)
    latency_ms: float
    model_version: str
    # New optional fields — existing clients ignore them safely.
    shape_analysis: dict[str, Any] | None = None
    overlay_b64: str | None = Field(
        default=None,
        description="Contour overlay PNG (base64), color-coded by risk level",
    )


def get_session() -> "ort.InferenceSession":
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
    return _session


def preprocess(image: Image.Image) -> np.ndarray:
    """Grayscale → resize 256×256 → normalize [0, 1] → NCHW float32."""
    image = image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr[None, None, ...]  # (1, 1, H, W)


def postprocess(logits: np.ndarray) -> np.ndarray:
    """Sigmoid → threshold 0.5 → uint8 mask {0, 255}."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs[0, 0] > 0.5).astype(np.uint8) * 255


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _session is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        img_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    t0 = time.perf_counter()
    inp = preprocess(image)
    try:
        logits = get_session().run(None, {"input": inp})[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    mask = postprocess(logits)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Post-segmentation morphology (additive — never breaks core fields).
    shape_analysis = None
    overlay_b64 = None
    try:
        resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        result = analyze_from_pil(mask, resized)
        shape_analysis = result.get("shape_analysis")
        overlay_b64 = result.get("overlay_b64")
    except Exception as exc:
        # Analysis must not fail the prediction.
        print(f"shape analysis skipped: {exc}")

    return PredictResponse(
        mask_b64=mask_b64,
        latency_ms=round(latency_ms, 2),
        model_version=MODEL_VERSION,
        shape_analysis=shape_analysis,
        overlay_b64=overlay_b64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
