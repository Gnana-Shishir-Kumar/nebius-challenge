"""Nebius Job N3 — export the trained U-Net to ONNX with optional fp16 and parity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from jobs/export/ directly during local dev
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import UNet2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EndoSeg U-Net to ONNX")
    p.add_argument("--checkpoint", default="checkpoints/unet_best.pth",
                   help="Path to .pth checkpoint file")
    p.add_argument("--out-dir", default="browser/model",
                   help="Directory to write unet.onnx into")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version")
    p.add_argument("--fp16", action="store_true",
                   help="Convert exported model weights to fp16")
    p.add_argument("--validate", action="store_true",
                   help="Run parity check: assert PyTorch vs ONNX MAE < 0.01")
    return p.parse_args()


def load_model(checkpoint: str) -> UNet2D:
    path = Path(checkpoint)
    if not path.exists():
        sys.exit(f"Checkpoint not found: {path}")
    model = UNet2D.for_browser()
    state = torch.load(str(path), map_location="cpu")
    weights = state.get("model", state)
    model.load_state_dict(weights)
    model.eval()
    print(f"Loaded checkpoint: {path}  ({model.model_size_mb():.1f} MB params)")
    return model


def export_onnx(model: UNet2D, out_path: Path, opset: int) -> None:
    dummy = torch.zeros(1, 1, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,  # force legacy exporter; new dynamo path is not ONNX Runtime Web compatible
    )
    print(f"Exported fp32 ONNX  -> {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


def convert_fp16(onnx_path: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(onnx_path))
    print(f"Converted to fp16   -> {onnx_path}  ({onnx_path.stat().st_size / 1e6:.2f} MB)")


def validate(model: UNet2D, onnx_path: Path) -> None:
    import onnxruntime as ort

    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        pt_out = model(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy()})[0]

    mae = float(np.abs(pt_out - ort_out).mean())
    assert mae < 0.01, f"Parity check FAILED: MAE={mae:.6f} >= 0.01"
    print(f"Parity check passed: MAE={mae:.6f}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)

    onnx_path = out_dir / "unet.onnx"
    export_onnx(model, onnx_path, args.opset)

    if args.fp16:
        convert_fp16(onnx_path)

    if args.validate:
        validate(model, onnx_path)

    print(f"\nFinal model: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
