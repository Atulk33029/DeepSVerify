"""
preprocessing/transforms.py

Phase 2 PILOT transforms. Everything here is deterministic (no randomness)
and works on plain numpy uint8 arrays. Nothing is thresholded into the
output: Otsu is used ONLY to locate the ink bounding box and to estimate
the background level; the stored pixels stay grayscale.

Pipeline for one image:
  1. load_grayscale     -> 8-bit grayscale (RGBA composited on white)
  2. detect_foreground  -> Otsu threshold, ink mask (pixel <= threshold),
                           ink bounding box
  3. optional intensity variant "flatten_bg": one global gain that maps
     the estimated background level to 255 (no thresholding). On
     already-white backgrounds (BHSig260) the gain is exactly 1.0, so no
     dataset-specific rule is needed.
  4. crop_with_margin   -> crop to bbox plus margin_frac of the bbox size
                           on each side (per axis), clamped to the image
  5. resize_and_pad     -> uniform scale (aspect preserved) to fit inside
                           the target canvas, centered, padded with
                           pad_value

Stored canonical image != model input normalization: no mean/std scaling
happens here; that belongs to the training/inference transform later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .eda import _otsu_threshold

# Every record produced by this module has exactly these keys, so metadata
# tables have a stable schema even for failed images.
RECORD_FIELDS = [
    "original_width", "original_height", "original_mode", "original_aspect_ratio",
    "alpha_composited", "alpha_min",
    "otsu_threshold", "foreground_fraction",
    "foreground_bbox_xyxy", "bbox_width", "bbox_height", "bbox_aspect_ratio",
    "margin_fraction", "margin_px_x", "margin_px_y", "margin_clamped", "crop_box_xyxy",
    "intensity_variant", "bg_level", "gain_applied",
    "scale_factor", "content_width", "content_height", "offset_x", "offset_y",
    "content_fill_fraction", "content_aspect_ratio", "aspect_distortion",
    "processed_width", "processed_height",
    "out_min", "out_max", "out_mean", "output_sha256",
    "preprocessing_version", "config_fingerprint",
    "preprocessing_status", "flags", "error",
]


def config_fingerprint(cfg: dict) -> str:
    keys = ("preprocessing_version", "target", "crop", "foreground",
            "intensity", "interpolation", "output", "flags")
    relevant = {k: cfg[k] for k in keys if k in cfg}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _empty_record(cfg: dict) -> dict:
    rec = {k: None for k in RECORD_FIELDS}
    rec["preprocessing_version"] = cfg.get("preprocessing_version")
    rec["config_fingerprint"] = config_fingerprint(cfg)
    rec["flags"] = ""
    return rec


# --------------------------------------------------------------------------
# 1. Loading / mode conversion
# --------------------------------------------------------------------------

def load_grayscale(path: Path) -> tuple[np.ndarray, dict]:
    """Returns (uint8 2D array, load_info). Consistent mode handling:
      - RGBA / LA / PA / P-with-transparency: composited onto WHITE, then L
      - 16-bit / 32-bit integer modes: linearly scaled to 8-bit
      - everything else (L, P, 1, RGB, ...): PIL convert("L")
    alpha_min < 255 means real transparency existed (flagged upstream).
    """
    with Image.open(path) as img:
        img.load()
        mode = img.mode
        width, height = img.size
        alpha_min = 255
        alpha_composited = False

        if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            alpha_min = int(np.asarray(rgba)[..., 3].min())
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            gray_img = Image.alpha_composite(background, rgba).convert("L")
            alpha_composited = True
        elif mode.startswith("I"):
            arr = np.asarray(img).astype(np.float64)
            scale = 1.0 / 257.0 if mode.startswith("I;16") else 1.0
            gray_img = Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8), mode="L")
        else:
            gray_img = img.convert("L")

    gray = np.array(gray_img, dtype=np.uint8)
    info = {
        "original_mode": mode,
        "original_width": width,
        "original_height": height,
        "alpha_composited": alpha_composited,
        "alpha_min": alpha_min,
    }
    return gray, info


# --------------------------------------------------------------------------
# 2. Foreground detection / background estimate
# --------------------------------------------------------------------------

def detect_foreground(gray: np.ndarray) -> dict:
    """Otsu threshold; ink = pixel <= threshold (see the '<=' fix in eda.py
    for why '<' breaks on bilevel images). Returns threshold, ink fraction,
    and bbox (x0, y0, x1, y1 inclusive) or None when no ink pixels exist.
    """
    threshold = int(_otsu_threshold(gray))
    mask = gray <= threshold
    fraction = float(mask.mean())
    bbox = None
    if mask.any():
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        bbox = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
    return {"threshold": threshold, "fraction": fraction, "bbox": bbox}


def estimate_background_level(gray: np.ndarray, threshold: int) -> Optional[float]:
    """Median of pixels above the Otsu threshold (i.e. the paper). None if
    there are no such pixels."""
    background = gray[gray > threshold]
    if background.size == 0:
        return None
    return float(np.median(background))


# --------------------------------------------------------------------------
# 3. Crop with margin
# --------------------------------------------------------------------------

def crop_with_margin(gray: np.ndarray, bbox: tuple, margin_frac: float):
    """Crop to bbox expanded by margin_frac * bbox_width horizontally and
    margin_frac * bbox_height vertically (each side), clamped to the image.
    Returns (crop, crop_box_xyxy, margin_px_x, margin_px_y, clamped)."""
    h, w = gray.shape
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    mx = int(round(margin_frac * bw))
    my = int(round(margin_frac * bh))
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(w - 1, x1 + mx), min(h - 1, y1 + my)
    clamped = (x0 - mx < 0) or (y0 - my < 0) or (x1 + mx > w - 1) or (y1 + my > h - 1)
    crop = gray[cy0:cy1 + 1, cx0:cx1 + 1]
    return crop, (cx0, cy0, cx1, cy1), mx, my, clamped


# --------------------------------------------------------------------------
# 4. Resize + pad (aspect ratio preserved)
# --------------------------------------------------------------------------

_RESAMPLE = {
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resize_and_pad(crop: np.ndarray, target_w: int, target_h: int, pad_value: int,
                   downscale: str = "box", upscale: str = "bicubic"):
    """Uniformly scale crop to fit inside (target_w, target_h), center it,
    fill the rest with pad_value. Never stretches an axis independently."""
    h, w = crop.shape
    scale = min(target_w / w, target_h / h)
    new_w = min(target_w, max(1, int(round(w * scale))))
    new_h = min(target_h, max(1, int(round(h * scale))))

    if new_w == w and new_h == h:
        resized = crop
    else:
        method = _RESAMPLE[downscale] if scale < 1 else _RESAMPLE[upscale]
        resized = np.asarray(Image.fromarray(crop, mode="L").resize((new_w, new_h), method))

    canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    off_x, off_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    info = {
        "scale_factor": float(scale),
        "content_width": new_w, "content_height": new_h,
        "offset_x": off_x, "offset_y": off_y,
        "content_fill_fraction": float(new_w * new_h / (target_w * target_h)),
        "content_aspect_ratio": float(new_w / new_h),
        "aspect_distortion": float(abs((new_w / new_h) / (w / h) - 1.0)),
    }
    return canvas, info


# --------------------------------------------------------------------------
# 5. Full single-image pipeline
# --------------------------------------------------------------------------

def _fmt_box(box) -> str:
    return "" if box is None else ",".join(str(int(v)) for v in box)


def preprocess_gray(gray: np.ndarray, load_info: dict, cfg: dict,
                    margin_frac: float, intensity_variant: str):
    """Runs steps 2-5 on an already-loaded grayscale array.
    Returns (canvas uint8 array, record dict). Difficult images are
    FLAGGED, never dropped: with no detectable ink the whole image is
    resized+padded instead."""
    rec = _empty_record(cfg)
    flags: list[str] = []
    fcfg, tcfg = cfg["flags"], cfg["target"]
    height, width = gray.shape

    rec.update({
        "original_width": load_info["original_width"],
        "original_height": load_info["original_height"],
        "original_mode": load_info["original_mode"],
        "original_aspect_ratio": load_info["original_width"] / load_info["original_height"],
        "alpha_composited": load_info["alpha_composited"],
        "alpha_min": load_info["alpha_min"],
        "margin_fraction": margin_frac,
        "intensity_variant": intensity_variant,
    })
    if load_info["alpha_min"] < 255:
        flags.append("alpha_nontrivial")

    det = detect_foreground(gray)
    rec["otsu_threshold"] = det["threshold"]
    rec["foreground_fraction"] = det["fraction"]
    bbox = det["bbox"]

    # Intensity variant (global gain only; no thresholding of stored pixels)
    working = gray
    bg_level = estimate_background_level(gray, det["threshold"])
    rec["bg_level"] = bg_level
    gain = 1.0
    if intensity_variant == "flatten_bg":
        if bg_level is not None and bg_level >= cfg["intensity"]["flatten_min_bg_level"]:
            gain = 255.0 / bg_level
            working = np.clip(np.round(gray.astype(np.float64) * gain), 0, 255).astype(np.uint8)
        else:
            flags.append("bg_level_unusual")
    elif intensity_variant != "original":
        raise ValueError(f"Unknown intensity_variant: {intensity_variant}")
    rec["gain_applied"] = gain

    if det["fraction"] > fcfg["polarity_suspect_fraction"]:
        flags.append("polarity_suspect")

    if bbox is None:
        flags.append("no_foreground")
        crop = working
        crop_box, mx, my, clamped = (0, 0, width - 1, height - 1), 0, 0, False
    else:
        bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
        rec.update({"foreground_bbox_xyxy": _fmt_box(bbox), "bbox_width": bw,
                    "bbox_height": bh, "bbox_aspect_ratio": bw / bh})
        if det["fraction"] < fcfg["tiny_foreground_fraction"]:
            flags.append("tiny_foreground")
        if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1:
            flags.append("bbox_touches_border")
        if bw / bh > fcfg["extreme_aspect_high"] or bw / bh < fcfg["extreme_aspect_low"]:
            flags.append("extreme_aspect")
        crop, crop_box, mx, my, clamped = crop_with_margin(working, bbox, margin_frac)
        if clamped:
            flags.append("margin_clamped")

    rec.update({"margin_px_x": mx, "margin_px_y": my, "margin_clamped": clamped,
                "crop_box_xyxy": _fmt_box(crop_box)})

    interp = cfg["interpolation"]
    canvas, info = resize_and_pad(crop, tcfg["width"], tcfg["height"], tcfg["pad_value"],
                                  interp["downscale"], interp["upscale"])
    rec.update(info)
    if info["scale_factor"] > fcfg["heavy_upscale_factor"]:
        flags.append("heavy_upscale")

    if canvas.min() == canvas.max():
        flags.append("blank_output")

    rec.update({
        "processed_width": int(canvas.shape[1]), "processed_height": int(canvas.shape[0]),
        "out_min": int(canvas.min()), "out_max": int(canvas.max()),
        "out_mean": float(canvas.mean()),
        "output_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
        "flags": ";".join(flags),
        "preprocessing_status": "flagged" if flags else "ok",
    })
    return canvas, rec


def failed_record(cfg: dict, message: str) -> dict:
    rec = _empty_record(cfg)
    rec["preprocessing_status"] = "failed"
    rec["error"] = message
    return rec


def preprocess_image(path, cfg: dict, margin_frac: float, intensity_variant: str):
    """Load + preprocess one file. Never raises: missing/corrupt files come
    back as (None, failed_record) so callers can count and report them."""
    p = Path(path)
    if not p.exists():
        return None, failed_record(cfg, "file_not_found")
    try:
        gray, load_info = load_grayscale(p)
    except Exception as exc:  # unreadable / corrupt
        return None, failed_record(cfg, f"unreadable: {exc}")
    try:
        return preprocess_gray(gray, load_info, cfg, margin_frac, intensity_variant)
    except Exception as exc:
        return None, failed_record(cfg, f"processing_error: {exc}")
