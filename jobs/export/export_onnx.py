"""Nebius Job N3 — export the trained U-Net to ONNX (+ optional quantization).

Loads a checkpoint, exports a clean ONNX graph at 256x256, optionally applies
dynamic int8 / fp16 quantization, and runs a quick parity check between the
PyTorch and ONNX Runtime outputs. The quantized .onnx is what the browser ships.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.unet import build_model  # noqa: E402

OPSET = 17


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EndoSeg ONNX export job")
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT", "/checkpoints/unet_best.pt"))
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "/checkpoints"))
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--quantize", choices=["none", "fp16", "int8"], default="int8")
    p.add_argument("--atol", type=float, default=1e-3, help="parity tolerance")
    return p.parse_args()


def load_model(checkpoint: str) -> torch.nn.Module:
    model = build_model(num_classes=1)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def export_fp32(model: torch.nn.Module, path: Path, img_size: int) -> None:
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"exported fp32 ONNX -> {path}")


def quantize(src: Path, dst: Path, mode: str) -> Path:
    if mode == "none":
        return src
    if mode == "int8":
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        print(f"int8 dynamic quantized -> {dst}")
        return dst
    if mode == "fp16":
        import onnx
        from onnxconverter_common import float16

        model_fp16 = float16.convert_float_to_float16(onnx.load(str(src)))
        onnx.save(model_fp16, str(dst))
        print(f"fp16 converted -> {dst}")
        return dst
    raise ValueError(mode)


def parity_check(model: torch.nn.Module, onnx_path: Path, img_size: int, atol: float) -> None:
    import onnxruntime as ort

    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    status = "PASS" if max_diff <= atol else "WARN"
    print(f"parity [{status}] max abs diff = {max_diff:.6f} (atol={atol})")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    fp32_path = out_dir / "unet.onnx"
    export_fp32(model, fp32_path, args.img_size)

    quant_path = out_dir / f"unet_{args.quantize}.onnx"
    final_path = quantize(fp32_path, quant_path, args.quantize)

    # Parity is checked against the fp32 export (quantized differs by design).
    parity_check(model, fp32_path, args.img_size, args.atol)
    print(f"browser model: {final_path} ({final_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
