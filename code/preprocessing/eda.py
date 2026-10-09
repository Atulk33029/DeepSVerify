"""
preprocessing/eda.py

Phase 2, Part 1 -- descriptive statistics on the REAL Phase 1 unified
metadata (metadata.csv). No synthetic data, no assumptions about the
dataset layout: everything here reads directly from the artifacts Phase 1
already produced.

Two kinds of analysis:

1. Metadata-only statistics (`compute_metadata_stats`): dimensions,
   aspect ratios, modes, extensions, file sizes, per-dataset and
   per-signature-type breakdowns. Fast -- uses only the columns already
   in metadata.csv, no image loading.

2. Sampled pixel-level statistics (`sample_pixel_stats`): for a small,
   deterministic, per-dataset sample of real images, loads the actual
   pixel data and computes simple foreground/background descriptors
   (mean/std intensity, an Otsu-threshold foreground bounding box, and
   the foreground area fraction). This is deliberately a SAMPLE, not a
   full pass over 16,680 images -- it exists purely to give real,
   measured evidence for the Part 3 preprocessing-strategy decision
   (per the project's "inspect real images before choosing a method"
   rule), not to preprocess anything yet.

Both functions return plain dicts (JSON-serializable) so results can be
printed, saved, and pasted back for review.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


# --------------------------------------------------------------------------
# Part 1: metadata-only statistics
# --------------------------------------------------------------------------

def load_metadata(metadata_csv_path: Path) -> pd.DataFrame:
    """Load Phase 1's metadata.csv. Coerces width/height to numeric,
    leaving non-numeric entries (the literal string 'unknown' for invalid
    images) as NaN rather than raising or silently dropping rows.
    """
    df = pd.read_csv(metadata_csv_path)
    for col in ("width", "height", "file_size_bytes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _describe(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()) if s.count() > 1 else 0.0,
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


def compute_metadata_stats(df: pd.DataFrame) -> dict:
    """Descriptive statistics computable from metadata.csv alone --
    dimensions, aspect ratios, modes, extensions, file sizes, and
    per-dataset / per-signature-type breakdowns.
    """
    valid = df[df["is_valid"] == True].copy()  # noqa: E712 (explicit bool compare on mixed-type CSV column)
    valid["aspect_ratio"] = valid["width"] / valid["height"]

    result = {
        "overall": {
            "total_rows": int(len(df)),
            "valid_rows": int(len(valid)),
            "width": _describe(valid["width"]),
            "height": _describe(valid["height"]),
            "aspect_ratio": _describe(valid["aspect_ratio"]),
            "file_size_bytes": _describe(valid["file_size_bytes"]),
            "modes": valid["mode"].value_counts(dropna=False).to_dict(),
            "extensions": valid["extension"].value_counts(dropna=False).to_dict(),
        },
        "by_dataset": {},
        "by_dataset_and_signature_type": {},
    }

    for dataset, group in valid.groupby("dataset"):
        result["by_dataset"][dataset] = {
            "count": int(len(group)),
            "width": _describe(group["width"]),
            "height": _describe(group["height"]),
            "aspect_ratio": _describe(group["aspect_ratio"]),
            "file_size_bytes": _describe(group["file_size_bytes"]),
            "modes": group["mode"].value_counts(dropna=False).to_dict(),
            "extensions": group["extension"].value_counts(dropna=False).to_dict(),
        }

    for (dataset, sig_type), group in valid.groupby(["dataset", "signature_type"]):
        key = f"{dataset}__{sig_type}"
        result["by_dataset_and_signature_type"][key] = {
            "count": int(len(group)),
            "width": _describe(group["width"]),
            "height": _describe(group["height"]),
            "aspect_ratio": _describe(group["aspect_ratio"]),
        }

    return result


def print_metadata_stats(stats: dict) -> None:
    """Human-readable console printout of compute_metadata_stats output."""
    o = stats["overall"]
    print("=" * 60)
    print("PHASE 2 / PART 1: METADATA-ONLY STATISTICS")
    print("=" * 60)
    print(f"Total rows: {o['total_rows']}  |  Valid rows: {o['valid_rows']}")
    print(f"Width  -- min={o['width'].get('min')} max={o['width'].get('max')} "
          f"mean={o['width'].get('mean'):.1f} median={o['width'].get('median')} "
          f"p05={o['width'].get('p05')} p95={o['width'].get('p95')}")
    print(f"Height -- min={o['height'].get('min')} max={o['height'].get('max')} "
          f"mean={o['height'].get('mean'):.1f} median={o['height'].get('median')} "
          f"p05={o['height'].get('p05')} p95={o['height'].get('p95')}")
    ar = o["aspect_ratio"]
    print(f"Aspect ratio (w/h) -- min={ar.get('min'):.2f} max={ar.get('max'):.2f} "
          f"mean={ar.get('mean'):.2f} median={ar.get('median'):.2f} "
          f"p05={ar.get('p05'):.2f} p95={ar.get('p95'):.2f}")
    print(f"Modes: {o['modes']}")
    print(f"Extensions: {o['extensions']}")
    fsz = o["file_size_bytes"]
    if fsz.get("count", 0):
        print(f"File size (bytes) -- mean={fsz['mean']:.0f} median={fsz['median']:.0f} "
              f"min={fsz['min']:.0f} max={fsz['max']:.0f}")

    print("\n--- Per dataset ---")
    for name, d in stats["by_dataset"].items():
        print(f"\n{name}: {d['count']} images")
        print(f"  width  mean={d['width'].get('mean', float('nan')):.1f}  "
              f"height mean={d['height'].get('mean', float('nan')):.1f}  "
              f"aspect_ratio mean={d['aspect_ratio'].get('mean', float('nan')):.2f}")
        print(f"  modes: {d['modes']}  extensions: {d['extensions']}")

    print("\n--- Per dataset x signature_type ---")
    for key, d in stats["by_dataset_and_signature_type"].items():
        print(f"  {key}: n={d['count']}  "
              f"width_mean={d['width'].get('mean', float('nan')):.1f}  "
              f"height_mean={d['height'].get('mean', float('nan')):.1f}  "
              f"aspect_ratio_mean={d['aspect_ratio'].get('mean', float('nan')):.2f}")


# --------------------------------------------------------------------------
# Part 2: sampled pixel-level foreground/background statistics
# --------------------------------------------------------------------------

def _otsu_threshold(gray: np.ndarray) -> int:
    """Minimal Otsu's method implementation (no OpenCV/skimage dependency).
    gray: uint8 2D array. Returns the threshold in [0, 255].
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg, max_between, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = t
    return threshold


def analyze_single_image(path: Path) -> Optional[dict]:
    """Loads one real image and computes simple, honest foreground/
    background descriptors:
      - grayscale mean/std intensity (background level + contrast)
      - an Otsu threshold and the resulting foreground pixel fraction
        (assumes darker-than-background = signature ink, which holds for
        all three datasets' typical white-paper scans; if the fraction
        comes back > 0.5 for a given image, that assumption likely
        doesn't hold for it, and this is flagged rather than silently
        trusted)
      - a foreground bounding box (min/max row/col of thresholded pixels)
        and its width/height/aspect ratio
    Returns None if the image cannot be opened.
    """
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None

    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    mean_intensity = float(arr.mean())
    std_intensity = float(arr.std())

    thresh = _otsu_threshold(arr)
    # Foreground = pixels at or below the threshold. Using strict "<" here
    # silently produces zero foreground whenever Otsu picks threshold=0 --
    # which happens on already-bilevel {0,255} images (confirmed on real
    # BHSig260 samples: 0 is a valid, correct separating point for such a
    # histogram, but nothing in a uint8 array is strictly less than 0, so
    # the mask wrongly captured no pixels at all even though 0 is exactly
    # the ink value). "<=" matches how the threshold was actually computed.
    fg_mask = arr <= thresh  # darker-than-or-equal-to-threshold = ink, on a light background
    fg_fraction = float(fg_mask.mean())

    bbox = None
    bbox_w = bbox_h = bbox_aspect = None
    assumption_flag = fg_fraction > 0.5  # background/foreground polarity may be inverted
    if fg_mask.any():
        rows = np.any(fg_mask, axis=1)
        cols = np.any(fg_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = (int(cmin), int(rmin), int(cmax), int(rmax))
        bbox_w, bbox_h = int(cmax - cmin + 1), int(rmax - rmin + 1)
        bbox_aspect = bbox_w / bbox_h if bbox_h else None

    return {
        "path": str(path),
        "image_width": w,
        "image_height": h,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "otsu_threshold": thresh,
        "foreground_fraction": fg_fraction,
        "foreground_polarity_suspect": assumption_flag,
        "foreground_bbox_xyxy": bbox,
        "foreground_bbox_width": bbox_w,
        "foreground_bbox_height": bbox_h,
        "foreground_bbox_aspect_ratio": bbox_aspect,
        "whitespace_fraction_removed_if_cropped": (
            1.0 - (bbox_w * bbox_h) / (w * h) if bbox_w and bbox_h else None
        ),
    }


def sample_pixel_stats(df: pd.DataFrame, n_per_group: int = 15, seed: int = 42) -> dict:
    """Deterministically samples up to n_per_group real, valid images from
    each (dataset, signature_type) combination, analyzes each with
    analyze_single_image, and aggregates the results. Uses `original_path`
    from metadata.csv, so this must be run in the same environment where
    those paths are valid (i.e. your Colab session with Drive mounted).

    Images that fail to open (moved/renamed since Phase 1 ran) are
    skipped and counted, not silently ignored.
    """
    rng = random.Random(seed)
    valid = df[df["is_valid"] == True]  # noqa: E712

    per_group_results = {}
    failures = []

    for (dataset, sig_type), group in valid.groupby(["dataset", "signature_type"]):
        paths = sorted(group["original_path"].tolist())  # sort first for determinism
        rng2 = random.Random(f"{seed}:{dataset}:{sig_type}")
        sampled = paths if len(paths) <= n_per_group else rng2.sample(paths, n_per_group)

        analyses = []
        for p in sampled:
            result = analyze_single_image(Path(p))
            if result is None:
                failures.append(p)
            else:
                analyses.append(result)

        key = f"{dataset}__{sig_type}"
        if analyses:
            per_group_results[key] = {
                "n_sampled": len(sampled),
                "n_analyzed": len(analyses),
                "mean_intensity": _describe(pd.Series([a["mean_intensity"] for a in analyses])),
                "std_intensity": _describe(pd.Series([a["std_intensity"] for a in analyses])),
                "foreground_fraction": _describe(pd.Series([a["foreground_fraction"] for a in analyses])),
                "foreground_bbox_aspect_ratio": _describe(
                    pd.Series([a["foreground_bbox_aspect_ratio"] for a in analyses if a["foreground_bbox_aspect_ratio"]])
                ),
                "whitespace_fraction_removed_if_cropped": _describe(
                    pd.Series([a["whitespace_fraction_removed_if_cropped"] for a in analyses
                               if a["whitespace_fraction_removed_if_cropped"] is not None])
                ),
                "n_polarity_suspect": sum(1 for a in analyses if a["foreground_polarity_suspect"]),
                "sample_paths": [a["path"] for a in analyses[:5]],
            }
        else:
            per_group_results[key] = {"n_sampled": len(sampled), "n_analyzed": 0}

    return {
        "seed": seed,
        "n_per_group": n_per_group,
        "groups": per_group_results,
        "n_failed_to_open": len(failures),
        "failed_paths_sample": failures[:10],
    }


def print_pixel_stats(stats: dict) -> None:
    print("=" * 60)
    print("PHASE 2 / PART 1 (cont.): SAMPLED PIXEL-LEVEL STATISTICS")
    print(f"(seed={stats['seed']}, up to {stats['n_per_group']} images per dataset x signature_type group)")
    print("=" * 60)
    if stats["n_failed_to_open"]:
        print(f"WARNING: {stats['n_failed_to_open']} sampled images failed to open "
              f"(sample of failed paths: {stats['failed_paths_sample']})")

    for key, g in stats["groups"].items():
        if g.get("n_analyzed", 0) == 0:
            print(f"\n{key}: no images successfully analyzed")
            continue
        print(f"\n{key}  (n_analyzed={g['n_analyzed']})")
        mi, si, ff = g["mean_intensity"], g["std_intensity"], g["foreground_fraction"]
        print(f"  mean_intensity: mean={mi['mean']:.1f} (0=black,255=white), std_across_images={mi['std']:.1f}")
        print(f"  std_intensity (contrast): mean={si['mean']:.1f}")
        print(f"  foreground_fraction (ink pixels / total): mean={ff['mean']:.3f} "
              f"min={ff['min']:.3f} max={ff['max']:.3f}")
        if g.get("n_polarity_suspect", 0):
            print(f"  WARNING: {g['n_polarity_suspect']} images had foreground_fraction > 0.5 "
                  f"-- polarity assumption (dark ink / light background) may not hold, inspect these")
        wr = g.get("whitespace_fraction_removed_if_cropped")
        if wr and wr.get("count"):
            print(f"  whitespace that a tight crop would remove: mean={wr['mean']:.1%}")
        bx = g.get("foreground_bbox_aspect_ratio")
        if bx and bx.get("count"):
            print(f"  foreground bbox aspect ratio: mean={bx['mean']:.2f} p05={bx['p05']:.2f} p95={bx['p95']:.2f}")
