"""
preprocessing/visual_audit.py

Phase 2, Part 7 -- builds a deterministic (fixed-seed) visual audit set
covering every required category:
  - CEDAR genuine/forgery, Bengali genuine/forgery, Hindi genuine/forgery
  - difficult cases: largest/smallest signature, thin strokes, large
    whitespace, unusual aspect ratio, images from duplicate groups,
    images flagged by QC

Selection is entirely metadata-driven (from preprocessing_metadata.csv +
the Phase 2 duplicate audit CSV), so it's reproducible without
re-scanning images. Grid rendering (original vs processed, side by side)
is a separate step so selection logic can be tested without image I/O.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def select_typical_examples(proc_df: pd.DataFrame, seed: int, n_per_group: int = 2) -> dict:
    """One or more typical (status=ok, no flags) examples per
    dataset x signature_type -- the required 6 baseline categories.
    """
    ok = proc_df[(proc_df["preprocessing_status"] == "ok") & (proc_df["flags"].fillna("") == "")]
    selected = {}
    for (dataset, sig_type), group in ok.groupby(["dataset", "signature_type"]):
        key = f"typical__{dataset}__{sig_type}"
        rng = random.Random(f"{seed}:{key}")
        ids = sorted(group["image_id"].tolist())
        chosen = ids if len(ids) <= n_per_group else rng.sample(ids, n_per_group)
        selected[key] = chosen
    return selected


def select_difficult_examples(proc_df: pd.DataFrame, dup_audit_df: pd.DataFrame | None, seed: int) -> dict:
    """Metadata-driven selection of the required 'difficult case'
    categories. Each category degrades gracefully to an empty list if no
    matching image exists in this batch, rather than raising.
    """
    rng = random.Random(seed)
    ok_or_flagged = proc_df[proc_df["preprocessing_status"].isin(["ok", "flagged"])].copy()
    selected = {}

    def _top(df: pd.DataFrame, by: str, n: int, ascending: bool = False) -> list:
        d = df.dropna(subset=[by])
        if d.empty:
            return []
        d = d.sort_values(by, ascending=ascending)
        return d["image_id"].head(n).tolist()

    # Largest / smallest actual signature content (by bbox area)
    bbox_area = ok_or_flagged["bbox_width"].astype(float) * ok_or_flagged["bbox_height"].astype(float)
    ok_or_flagged = ok_or_flagged.assign(_bbox_area=bbox_area)
    selected["largest_signature"] = _top(ok_or_flagged, "_bbox_area", 3, ascending=False)
    selected["smallest_signature"] = _top(ok_or_flagged, "_bbox_area", 3, ascending=True)

    # Thinnest strokes -> lowest foreground_fraction
    selected["thin_strokes"] = _top(ok_or_flagged, "foreground_fraction", 3, ascending=True)

    # Large whitespace -> lowest content_fill_fraction
    selected["large_whitespace"] = _top(ok_or_flagged, "content_fill_fraction", 3, ascending=True)

    # Unusual aspect ratio -> flagged extreme_aspect, else the most extreme bbox_aspect_ratio
    extreme_flagged = ok_or_flagged[ok_or_flagged["flags"].fillna("").str.contains("extreme_aspect")]
    if len(extreme_flagged):
        selected["unusual_aspect_ratio"] = sorted(extreme_flagged["image_id"].tolist())[:3]
    else:
        ar = ok_or_flagged["bbox_aspect_ratio"].astype(float)
        most_extreme = (ar - ar.median()).abs().sort_values(ascending=False)
        selected["unusual_aspect_ratio"] = ok_or_flagged.loc[most_extreme.head(3).index, "image_id"].tolist()

    # Unusual background -> alpha compositing applied, or an unusually bright/dark output
    alpha_flagged = ok_or_flagged[ok_or_flagged["flags"].fillna("").str.contains("alpha_nontrivial")]
    if len(alpha_flagged):
        selected["unusual_background"] = sorted(alpha_flagged["image_id"].tolist())[:3]
    else:
        om = ok_or_flagged["out_mean"].astype(float)
        most_extreme = (om - om.median()).abs().sort_values(ascending=False)
        selected["unusual_background"] = ok_or_flagged.loc[most_extreme.head(3).index, "image_id"].tolist()

    # Images from duplicate groups (Phase 2 Part 2 audit)
    if dup_audit_df is not None and len(dup_audit_df):
        exact = dup_audit_df[dup_audit_df["hash_type"] == "sha256_exact"]
        pool = sorted(exact["image_id"].dropna().unique().tolist())
        selected["duplicate_group_member"] = rng.sample(pool, min(3, len(pool))) if pool else []
    else:
        selected["duplicate_group_member"] = []

    # Images flagged by QC -- one example per distinct flag type present
    flagged_by_type = {}
    for flag in ["blank_output", "tiny_foreground", "extreme_aspect", "heavy_upscale",
                 "margin_clamped", "polarity_suspect", "no_foreground"]:
        mask = proc_df["flags"].fillna("").str.contains(flag)
        ids = sorted(proc_df.loc[mask, "image_id"].tolist())
        if ids:
            flagged_by_type[flag] = ids[0]
    selected["qc_flagged_examples"] = list(flagged_by_type.values())
    selected["qc_flagged_examples_by_type"] = flagged_by_type

    return selected


def build_audit_manifest(proc_df: pd.DataFrame, dup_audit_df: pd.DataFrame | None,
                         seed: int, n_per_typical_group: int = 2) -> dict:
    """Combines typical + difficult selections into one manifest dict:
    {category_name: [image_id, ...]}. Deterministic given the same inputs
    and seed.
    """
    manifest = {}
    manifest.update(select_typical_examples(proc_df, seed, n_per_typical_group))
    difficult = select_difficult_examples(proc_df, dup_audit_df, seed)
    for k, v in difficult.items():
        if k == "qc_flagged_examples_by_type":
            continue  # kept only for grid labeling, not a flat category
        manifest[k] = v
    return manifest


def render_before_after(image_id: str, original_path: str, processed_path: str | None,
                        category: str, out_path: Path) -> None:
    """Saves one PNG: original (resized to a consistent display height)
    next to the processed 384x128 output, labeled with the category.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    try:
        orig = Image.open(original_path).convert("L")
        axes[0].imshow(np.array(orig), cmap="gray", vmin=0, vmax=255)
    except Exception as exc:
        axes[0].text(0.5, 0.5, f"load error:\n{exc}", ha="center", va="center", fontsize=7)
    axes[0].set_title("original", fontsize=9)
    axes[0].axis("off")

    if processed_path:
        try:
            proc = Image.open(processed_path)
            axes[1].imshow(np.array(proc), cmap="gray", vmin=0, vmax=255)
        except Exception as exc:
            axes[1].text(0.5, 0.5, f"load error:\n{exc}", ha="center", va="center", fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "no output\n(failed)", ha="center", va="center")
    axes[1].set_title("processed (384x128)", fontsize=9)
    axes[1].axis("off")

    fig.suptitle(f"{category}: {image_id}", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_audit_manifest(manifest: dict, proc_df: pd.DataFrame, output_root: Path) -> list:
    """Renders every image in the manifest, organized under
    output_root/<category>/<image_id>.png. Returns the list of written
    paths. Skips (with a note) any image_id not found in proc_df rather
    than raising.
    """
    lookup = proc_df.set_index("image_id")[["original_path", "processed_path"]].to_dict("index")
    written = []
    for category, image_ids in manifest.items():
        for image_id in image_ids:
            info = lookup.get(image_id)
            if info is None:
                continue
            out_path = output_root / category / f"{image_id}.png"
            render_before_after(image_id, info["original_path"], info["processed_path"], category, out_path)
            written.append(out_path)
    return written
