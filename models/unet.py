"""Compact 2D U-Net for browser-friendly ovarian-lesion segmentation.

Kept intentionally small so the exported ONNX graph stays clean and the
quantized model fits the ~50 MB browser budget (see PRD §8). Default config
targets a 256x256 single-channel (grayscale) ultrasound input and a binary
lesion mask output (logits; apply sigmoid outside the graph).

ONNX-export notes:
- No in-place ops (``ReLU(inplace=False)``) so autograd/export stay clean.
- Decoder uses bilinear ``nn.Upsample`` (exports to a standard ``Resize`` node)
  instead of ``ConvTranspose2d`` — avoids checkerboard artefacts and keeps the
  graph free of custom/unsupported ops for ONNX Runtime Web.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x2 block used throughout the U-Net."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool downsample followed by a DoubleConv encoder stage."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample, concatenate the skip feature, then DoubleConv.

    Upsampling does not change the channel count, so the following DoubleConv
    receives ``in_ch + skip_ch`` channels and reduces them to ``out_ch``.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Minimal symmetric U-Net for binary segmentation.

    Encoder channels are ``[c, 2c, 4c, 8c]`` and the bottleneck is ``16c``.
    With ``base_channels=16`` this yields the target config:
    encoder ``[16, 32, 64, 128]`` and bottleneck ``256``.

    Args:
        in_channels: input image channels (1 for grayscale ultrasound).
        num_classes: output channels. 1 => binary (sigmoid) lesion mask.
        base_channels: channel width of the first encoder stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c = base_channels

        # Encoder: first stage has no pooling; the next three downsample.
        self.enc1 = DoubleConv(in_channels, c)        # 16
        self.enc2 = Down(c, c * 2)                    # 32
        self.enc3 = Down(c * 2, c * 4)                # 64
        self.enc4 = Down(c * 4, c * 8)                # 128

        # Bottleneck: 256
        self.bottleneck = Down(c * 8, c * 16)

        # Decoder: bilinear upsample + skip connection, 4 stages.
        self.up4 = Up(c * 16, c * 8, c * 8)           # 256 + 128 -> 128
        self.up3 = Up(c * 8, c * 4, c * 4)            # 128 +  64 ->  64
        self.up2 = Up(c * 4, c * 2, c * 2)            #  64 +  32 ->  32
        self.up1 = Up(c * 2, c, c)                    #  32 +  16 ->  16

        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.head(d1)  # logits; apply sigmoid outside the graph


def build_model(
    num_classes: int = 1,
    in_channels: int = 1,
    base_channels: int = 16,
) -> UNet:
    """Factory used by training/export so config stays in one place."""
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )


class UNet2D(UNet):
    """UNet with browser-oriented factory and size utilities.

    All existing ``UNet`` kwargs work; this class just adds two helpers that
    the public API and the verification script expect.
    """

    @classmethod
    def for_browser(cls) -> "UNet2D":
        """Browser-optimised default: 1-ch grayscale in, binary mask out, base=16."""
        return cls(in_channels=1, num_classes=1, base_channels=16)

    def model_size_mb(self) -> float:
        """Parameter footprint in MB (float32 — before quantization)."""
        return sum(p.numel() for p in self.parameters()) * 4 / (1024 ** 2)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(1, 1, 256, 256)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input shape : {tuple(dummy.shape)}")
    print(f"output shape: {tuple(out.shape)} | params: {n_params/1e6:.2f}M")
