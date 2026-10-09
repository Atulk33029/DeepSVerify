"""
preprocessing/pilot.py

Runs the PILOT comparison: a small, deterministic, stratified sample of
real images processed under all combinations of the two still-open
variables (intensity handling x margin fraction), so the decision between
them can be made from actual output rather than assumption.

This is intentionally separate from the (not-yet-written) full-batch
Part 3/4 pipeline runner -- outputs live under reports/phase2_pilot/ and
reports/phase2_visual_audit/, never under preprocessed/.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .transforms import preprocess_image, RECORD_FIELDS


def select_pilot_sample(df: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    """Deterministic stratified sample: up to n_per_group real, valid rows
    from each (dataset, signature_type) group. Sorted before sampling so
    the same input always yields the same sample.
    """
    valid = df[df["is_valid"] == True]  # noqa: E712
    parts = []
    for (dataset, sig_type), group in valid.groupby(["dataset", "signature_type"]):
        group_sorted = group.sort_values("image_id")
        rng = random.Random(f"{seed}:{dataset}:{sig_type}")
        idx = list(group_sorted.index)
        chosen = idx if len(idx) <= n_per_group else rng.sample(idx, n_per_group)
        parts.append(group_sorted.loc[sorted(chosen)])
    return pd.concat(parts, ignore_index=True)


def variant_name(intensity_variant: str, margin_frac: float) -> str:
    return f"{intensity_variant}_margin{int(round(margin_frac * 100)):02d}"


def run_pilot(sample_df: pd.DataFrame, cfg: dict, output_root: Path) -> pd.DataFrame:
    """Runs every (intensity_variant, margin_fraction) combination in
    cfg['pilot'] over sample_df, saving PNGs under
    output_root/<variant_name>/<image_id>.png and returning one combined
    metadata DataFrame (RECORD_FIELDS + image_id/dataset/writer/
    signature_type/variant/original_path columns).
    """
    pcfg = cfg["pilot"]
    rows = []

    for intensity_variant in pcfg["intensity_variants"]:
        for margin_frac in pcfg["margin_fractions"]:
            vname = variant_name(intensity_variant, margin_frac)
            variant_dir = output_root / vname
            variant_dir.mkdir(parents=True, exist_ok=True)

            for row in sample_df.itertuples(index=False):
                canvas, rec = preprocess_image(
                    row.original_path, cfg, margin_frac, intensity_variant
                )
                if canvas is not None:
                    out_path = variant_dir / f"{row.image_id}.png"
                    Image.fromarray(canvas, mode="L").save(out_path)
                    rec["output_path"] = str(out_path)
                else:
                    rec["output_path"] = None

                rec.update({
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "global_writer_id": row.global_writer_id,
                    "signature_type": row.signature_type,
                    "original_path": row.original_path,
                    "variant": vname,
                })
                rows.append(rec)

    columns = ["image_id", "dataset", "global_writer_id", "signature_type",
               "original_path", "variant", "output_path"] + RECORD_FIELDS
    return pd.DataFrame(rows)[columns]


def summarize_pilot(pilot_df: pd.DataFrame) -> pd.DataFrame:
    """Per-variant quantitative comparison: how often each variant's
    outputs are flagged, mean pixel-fill efficiency, mean aspect
    distortion, etc. -- the numeric complement to the visual grids.
    """
    def _pct(mask):
        return round(100.0 * mask.mean(), 1)

    summary = []
    for vname, g in pilot_df.groupby("variant"):
        ok = g[g["preprocessing_status"] == "ok"]
        summary.append({
            "variant": vname,
            "n_images": len(g),
            "n_failed": int((g["preprocessing_status"] == "failed").sum()),
            "n_flagged": int((g["preprocessing_status"] == "flagged").sum()),
            "pct_blank_output": _pct(g["flags"].fillna("").str.contains("blank_output")),
            "pct_tiny_foreground": _pct(g["flags"].fillna("").str.contains("tiny_foreground")),
            "pct_extreme_aspect": _pct(g["flags"].fillna("").str.contains("extreme_aspect")),
            "pct_heavy_upscale": _pct(g["flags"].fillna("").str.contains("heavy_upscale")),
            "pct_margin_clamped": _pct(g["flags"].fillna("").str.contains("margin_clamped")),
            "mean_content_fill_fraction": round(float(ok["content_fill_fraction"].mean()), 4) if len(ok) else None,
            "mean_aspect_distortion": round(float(ok["aspect_distortion"].mean()), 4) if len(ok) else None,
            "mean_out_mean_intensity": round(float(ok["out_mean"].mean()), 1) if len(ok) else None,
        })
    return pd.DataFrame(summary).sort_values("variant").reset_index(drop=True)


def print_pilot_summary(summary_df: pd.DataFrame) -> None:
    print("=" * 100)
    print("PHASE 2 PILOT: per-variant comparison")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("\ncontent_fill_fraction = fraction of the 384x128 canvas actually covered by signature "
          "content (higher = more efficient pixel use, less padding waste)")
    print("aspect_distortion = 0 means perfectly preserved aspect ratio (should always be ~0 here; "
          "a nonzero value would indicate a bug, not a variant tradeoff)")


def build_comparison_grid(image_id: str, original_path: str, variant_paths: dict, out_path: Path) -> None:
    """Saves a single PNG: original image + each variant's output, labeled,
    side by side. variant_paths: {variant_name: png_path_or_None}.
    """
    import matplotlib.pyplot as plt

    panels = [("original", original_path)] + list(variant_paths.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, panels):
        if path is None:
            ax.text(0.5, 0.5, "failed", ha="center", va="center")
        else:
            try:
                img = Image.open(path)
                ax.imshow(np.array(img), cmap="gray", vmin=0, vmax=255)
            except Exception as exc:
                ax.text(0.5, 0.5, f"error:\n{exc}", ha="center", va="center", fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    fig.suptitle(image_id)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def select_comparison_examples(sample_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Picks a small, deterministic, diverse subset to actually render
    full before/after grids for (one per dataset at minimum, plus extras).
    """
    rng = random.Random(seed)
    chosen_ids = []
    for dataset, group in sample_df.groupby("dataset"):
        idx = sorted(group.index)
        chosen_ids.append(rng.choice(idx))
    remaining_n = max(0, n - len(chosen_ids))
    remaining_pool = [i for i in sorted(sample_df.index) if i not in chosen_ids]
    if remaining_n and remaining_pool:
        chosen_ids += rng.sample(remaining_pool, min(remaining_n, len(remaining_pool)))
    return sample_df.loc[sorted(set(chosen_ids))]
