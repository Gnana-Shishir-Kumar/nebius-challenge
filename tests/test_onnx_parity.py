"""Parity test: PyTorch U-Net vs exported ONNX (fp32) under ONNX Runtime.

Guards the critical browser invariant — the model the browser runs must match
the trained PyTorch model within tolerance. Quantized models are expected to
diverge slightly, so parity is asserted against the fp32 export.

Run: pytest tests/test_onnx_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.unet import build_model  # noqa: E402

ort = pytest.importorskip("onnxruntime")

IMG_SIZE = 256
ATOL = 1e-3


@pytest.fixture(scope="module")
def exported_onnx(tmp_path_factory) -> tuple[torch.nn.Module, Path]:
    """Build a fresh model and export it to a temp ONNX file."""
    model = build_model(num_classes=1).eval()
    out_path = tmp_path_factory.mktemp("onnx") / "unet.onnx"
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    return model, out_path


def test_onnx_matches_pytorch(exported_onnx):
    model, onnx_path = exported_onnx
    x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        torch_out = model(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.numpy()})[0]

    assert onnx_out.shape == torch_out.shape
    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff <= ATOL, f"ONNX/PyTorch diverged: max abs diff {max_diff} > {ATOL}"


def test_output_shape(exported_onnx):
    _, onnx_path = exported_onnx
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.zeros((1, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)})[0]
    assert out.shape == (1, 1, IMG_SIZE, IMG_SIZE)
