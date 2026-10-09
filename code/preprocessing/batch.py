"""
preprocessing/batch.py

Phase 2, Parts 4/8/9/11 -- the full-batch runner. Processes every valid
image from the real Phase 1 metadata with the LOCKED configuration
(flatten_bg intensity, 10% margin, 384x128 canvas -- see
configs/preprocessing_config.yaml's `locked` section), using the same
preprocess_image() pipeline already unit-tested and pilot-reviewed.

Reproducibility (Part 11): deterministic processing order (sorted by
image_id), and RESUMABLE -- if preprocessing_metadata.csv already has a
row for an image_id with output_path pointing at an existing file AND
that row's config_fingerprint matches the current config, the image is
skipped rather than reprocessed. This makes it safe to re-run after a
Colab disconnect without accumulating duplicate work or stale files: any
image whose config changed, or whose output file is missing, gets
(re)processed; everything else is left untouched.

Provenance (Part 8): every output row is traceable back to its Phase 1
image_id, with only the Phase 1 fields explicitly required by the spec
duplicated here (dataset, writer_id, global_writer_id, signature_type,
original_path) -- everything else about an image's Phase 1 identity
stays in metadata.csv as the source of truth, joinable by image_id.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .transforms import preprocess_image, config_fingerprint

# Part 8's "at minimum include" list, plus a small set of technical
# fields that aid QC/traceability without duplicating Phase 1 content
# (otsu_threshold, bbox/content geometry, output hash, flags).
FINAL_METADATA_COLUMNS = [
    "image_id", "dataset", "writer_id", "global_writer_id", "signature_type",
    "original_path", "processed_path",
    "original_width", "original_height", "original_mode", "original_aspect_ratio",
    "processed_width", "processed_height",
    "preprocessing_version", "preprocessing_status",
    "foreground_bbox_xyxy", "bbox_width", "bbox_height", "bbox_aspect_ratio",
    "margin_fraction", "margin_px_x", "margin_px_y", "margin_clamped", "crop_box_xyxy",
    "intensity_variant", "bg_level", "gain_applied",
    "scale_factor", "content_width", "content_height", "content_fill_fraction", "aspect_distortion",
    "otsu_threshold", "foreground_fraction",
    "out_min", "out_max", "out_mean", "output_sha256",
    "config_fingerprint", "flags", "error",
]


def _load_existing(metadata_csv_path: Path) -> dict:
    if not metadata_csv_path.exists():
        return {}
    prev = pd.read_csv(metadata_csv_path)
    return {row["image_id"]: row.to_dict() for _, row in prev.iterrows()}


def run_full_batch(df: pd.DataFrame, cfg: dict, images_dir: Path, metadata_csv_path: Path,
                   resume: bool = True, progress_every: int = 500,
                   progress_cb=None) -> tuple[pd.DataFrame, dict]:
    """Processes every row where is_valid is True, in deterministic
    (sorted by image_id) order. Returns (metadata_df, run_stats).

    progress_cb(i, total, processed, skipped, failed), if given, is called
    every `progress_every` images (in addition to the default print), so
    a notebook can render a live progress bar without this module
    depending on any UI library.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(cfg)
    locked = cfg["locked"]
    margin_frac = locked["margin_fraction"]
    intensity_variant = locked["intensity_variant"]

    existing = _load_existing(metadata_csv_path) if resume else {}

    valid_df = df[df["is_valid"] == True].sort_values("image_id")  # noqa: E712
    total = len(valid_df)

    results = []
    processed_count = skipped_count = failed_count = 0

    for i, row in enumerate(valid_df.itertuples(index=False), start=1):
        image_id = row.image_id
        out_path = images_dir / f"{image_id}.png"
        prev = existing.get(image_id)

        can_skip = (
            resume and prev is not None
            and prev.get("config_fingerprint") == fingerprint
            and prev.get("margin_fraction") == margin_frac
            and prev.get("intensity_variant") == intensity_variant
            and prev.get("preprocessing_status") in ("ok", "flagged")
            and out_path.exists()
        )
        if can_skip:
            results.append(prev)
            skipped_count += 1
        else:
            canvas, rec = preprocess_image(row.original_path, cfg, margin_frac, intensity_variant)
            if canvas is not None:
                Image.fromarray(canvas, mode="L").save(out_path)
                rec["processed_path"] = str(out_path)
            else:
                rec["processed_path"] = None
                failed_count += 1
            rec.update({
                "image_id": image_id,
                "dataset": row.dataset,
                "writer_id": row.writer_id,
                "global_writer_id": row.global_writer_id,
                "signature_type": row.signature_type,
                "original_path": row.original_path,
            })
            results.append(rec)
            processed_count += 1

        if i % progress_every == 0 or i == total:
            print(f"  ... {i}/{total} images  "
                  f"(processed={processed_count}, resumed={skipped_count}, failed_so_far={failed_count})")
            if progress_cb is not None:
                progress_cb(i, total, processed_count, skipped_count, failed_count)

    out_df = pd.DataFrame(results)
    for col in FINAL_METADATA_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df[FINAL_METADATA_COLUMNS]
    out_df.to_csv(metadata_csv_path, index=False)

    stats = {
        "total_valid_input_images": total,
        "newly_processed": processed_count,
        "resumed_from_previous_run": skipped_count,
        "failed": failed_count,
        "output_rows_written": len(out_df),
    }
    return out_df, stats
