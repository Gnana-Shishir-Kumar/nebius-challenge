"""Post-segmentation morphology analysis (IOTA-inspired clinical hints).

Uses OpenCV contour geometry only — no diagnosis, research/education only.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from PIL import Image


# BGR contour colors by risk (OpenCV)
_RISK_BGR = {
    "low": (80, 180, 80),       # green
    "low-moderate": (0, 200, 220),  # yellow-ish
    "moderate": (0, 140, 255),  # orange
    "higher": (60, 60, 220),    # red
}


def _empty_analysis(reason: str = "No lesion contour found") -> dict[str, Any]:
    return {
        "shape_name": "Indeterminate",
        "risk_level": "low",
        "clinical_hint": reason,
        "circularity": 0.0,
        "solidity": 0.0,
        "aspect_ratio": 0.0,
        "lobe_count": 0,
        "area_px": 0,
    }


def _count_lobes(contour: np.ndarray) -> int:
    """Approx lobulation count via convexity defects (fallback: 1)."""
    if len(contour) < 5:
        return 1
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 1
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return 1
    if defects is None:
        return 1
    # OpenCV returns (N,1,4) or (N,4) depending on version; depth is fixed-point /256.
    defects = np.asarray(defects).reshape(-1, 4)
    significant = 0
    for start, end, farthest, depth_fp in defects:
        depth = float(depth_fp) / 256.0
        if depth > 2.0:
            significant += 1
    # lobes ≈ indentations + 1 for a closed outline
    return max(1, significant + 1)


def _classify_iota(
    circularity: float,
    solidity: float,
    aspect_ratio: float,
    lobe_count: int,
) -> tuple[str, str, str]:
    """Map geometry → (shape_name, risk_level, clinical_hint)."""
    # Multilocular / complex
    if lobe_count >= 5 or (solidity < 0.75 and lobe_count >= 3):
        return (
            "Multilocular / Complex",
            "higher",
            "Multiple lobulations or low solidity suggest a multilocular / complex outline "
            "(IOTA-inspired morphology). Further expert review recommended.",
        )
    # Irregular / lobulated
    if lobe_count >= 3 or solidity < 0.85 or circularity < 0.55:
        return (
            "Irregular / Lobulated",
            "moderate",
            "Irregular or lobulated contour with reduced circularity/solidity "
            "(IOTA-inspired). Morphology is non-smooth — interpret cautiously.",
        )
    # Round / unilocular
    if circularity >= 0.85 and solidity >= 0.90 and lobe_count <= 2 and aspect_ratio <= 1.35:
        return (
            "Round / Unilocular",
            "low",
            "Nearly circular, solid contour consistent with a simple unilocular outline "
            "(IOTA-inspired morphology).",
        )
    # Oval / regular
    return (
        "Oval / Regular",
        "low-moderate",
        "Elongated or mildly asymmetric but regular contour "
        "(IOTA-inspired oval / regular morphology).",
    )


def analyze_mask(mask: np.ndarray, original_rgb: np.ndarray | None = None) -> dict[str, Any]:
    """Analyze a binary mask (H×W, values 0/255) and optionally build overlay PNG b64.

    Returns dict with keys: shape_analysis (metrics+labels), overlay_b64 (str|None).
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    binary = (mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        analysis = _empty_analysis()
        return {"shape_analysis": analysis, "overlay_b64": None}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 16:
        analysis = _empty_analysis("Lesion area too small for reliable morphology")
        return {"shape_analysis": analysis, "overlay_b64": None}

    peri = float(cv2.arcLength(contour, True))
    circularity = float(4.0 * np.pi * area / (peri * peri)) if peri > 1e-6 else 0.0
    circularity = float(np.clip(circularity, 0.0, 1.5))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = float(np.clip(area / hull_area, 0.0, 1.0))

    if len(contour) >= 5:
        (_cx, _cy), (ma, MA), _angle = cv2.fitEllipse(contour)
        minor, major = sorted([float(ma), float(MA)])
        aspect_ratio = float(major / minor) if minor > 1e-6 else 1.0
    else:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(max(w, h) / max(1, min(w, h)))

    lobe_count = int(_count_lobes(contour))
    shape_name, risk_level, hint = _classify_iota(
        circularity, solidity, aspect_ratio, lobe_count
    )

    analysis = {
        "shape_name": shape_name,
        "risk_level": risk_level,
        "clinical_hint": hint,
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "lobe_count": lobe_count,
        "area_px": int(round(area)),
    }

    overlay_b64 = _draw_overlay(binary, contour, risk_level, original_rgb)
    return {"shape_analysis": analysis, "overlay_b64": overlay_b64}


def _draw_overlay(
    binary: np.ndarray,
    contour: np.ndarray,
    risk_level: str,
    original_rgb: np.ndarray | None,
) -> str | None:
    h, w = binary.shape[:2]
    if original_rgb is not None:
        base = original_rgb
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        elif base.shape[2] == 4:
            base = cv2.cvtColor(base, cv2.COLOR_RGBA2BGR)
        else:
            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        if base.shape[0] != h or base.shape[1] != w:
            base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas = base.copy()
    else:
        canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    color = _RISK_BGR.get(risk_level, _RISK_BGR["low"])
    cv2.drawContours(canvas, [contour], -1, color, 2)
    # light fill
    overlay = canvas.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def analyze_from_pil(mask_u8: np.ndarray, image: Image.Image) -> dict[str, Any]:
    """Convenience: mask ndarray + PIL original → analysis + overlay_b64."""
    rgb = np.asarray(image.convert("RGB").resize((mask_u8.shape[1], mask_u8.shape[0])))
    return analyze_mask(mask_u8, original_rgb=rgb)
