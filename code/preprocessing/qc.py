"""
preprocessing/qc.py

Phase 2, Part 6 -- automated quality control over the full-batch
preprocessing output. Never deletes or modifies anything; every check
either passes/fails or produces a list of flagged image_ids for human
review, per the project's "flag suspicious images instead of silently
deleting them" rule.

Checks are split into two tiers:
  - CRITICAL: structural correctness of the pipeline itself (counts
    match, no duplicate/missing outputs, dimensions correct, no aspect
    distortion, valid pixel range). Any failure here means something is
    actually broken and Phase 2 cannot be declared PASS.
  - INFORMATIONAL: legitimate per-image difficulty flags (blank output,
    tiny foreground, extreme aspect ratio, heavy upscale, margin
    clamped, polarity suspect). These are expected to occur at some rate
    on real data (confirmed during the pilot) and are reported with
    counts/rates for human review, not treated as pipeline failures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _flag_mask(df: pd.DataFrame, flag: str) -> pd.Series:
    return df["flags"].fillna("").astype(str).str.contains(flag, regex=False)


def run_full_batch_qc(phase1_df: pd.DataFrame, proc_df: pd.DataFrame, images_dir: Path, cfg: dict) -> dict:
    valid_ids = set(phase1_df.loc[phase1_df["is_valid"] == True, "image_id"])  # noqa: E712
    proc_ids = list(proc_df["image_id"])
    proc_id_set = set(proc_ids)

    duplicate_image_ids = proc_df["image_id"][proc_df["image_id"].duplicated()].tolist()
    missing_outputs = sorted(valid_ids - proc_id_set)
    unexpected_extra_rows = sorted(proc_id_set - valid_ids)

    on_disk = {p.stem for p in images_dir.glob("*.png")}
    referenced_on_disk = set(proc_df.loc[proc_df["processed_path"].notna(), "image_id"])
    missing_files_on_disk = sorted(referenced_on_disk - on_disk)
    extra_files_on_disk = sorted(on_disk - referenced_on_disk)

    ok_df = proc_df[proc_df["preprocessing_status"] == "ok"]
    flagged_df = proc_df[proc_df["preprocessing_status"] == "flagged"]
    failed_df = proc_df[proc_df["preprocessing_status"] == "failed"]

    target_w, target_h = cfg["target"]["width"], cfg["target"]["height"]
    ok_and_flagged = pd.concat([ok_df, flagged_df])
    dim_mismatch = ok_and_flagged[
        (ok_and_flagged["processed_width"] != target_w) | (ok_and_flagged["processed_height"] != target_h)
    ]
    pixel_range_bad = ok_and_flagged[
        (ok_and_flagged["out_min"] < 0) | (ok_and_flagged["out_max"] > 255)
    ]
    # resize_and_pad rounds its scaled content to integer pixel dimensions,
    # which introduces a small, mathematically bounded aspect "distortion"
    # purely from that rounding -- NOT a stretching bug. The worst case for
    # a single round() is a 0.5px shift, so the worst-case relative error
    # scales as ~0.5 / min(content_width, content_height): negligible for
    # typical crops resized to ~100+ px, but measurably higher (~3-6%) for
    # very elongated real content (confirmed possible on real BHSig260
    # ink-bbox aspect ratios from the Phase 2 EDA) whose smaller resized
    # dimension lands in the teens of pixels. A genuine independent-axis
    # stretch BUG is not bounded by this relationship at all -- it would
    # show distortion unrelated to how small the content is -- so each row
    # gets its own expected-noise bound (with a 2x safety margin and a 3%
    # floor) rather than one fixed constant that would either be too loose
    # for large content or wrongly fail legitimate thin content.
    def _distortion_bound(row) -> float:
        min_dim = min(row.get("content_width") or 1, row.get("content_height") or 1)
        return max(0.03, min(1.0, (0.5 / min_dim) * 2))

    distortion_bounds = ok_and_flagged.apply(_distortion_bound, axis=1)
    aspect_distortion_bug = ok_and_flagged[
        ok_and_flagged["aspect_distortion"].astype(float) > distortion_bounds
    ]

    informational = {
        "blank_output": _flag_mask(proc_df, "blank_output"),
        "tiny_foreground": _flag_mask(proc_df, "tiny_foreground"),
        "extreme_aspect": _flag_mask(proc_df, "extreme_aspect"),
        "heavy_upscale": _flag_mask(proc_df, "heavy_upscale"),
        "margin_clamped": _flag_mask(proc_df, "margin_clamped"),
        "polarity_suspect": _flag_mask(proc_df, "polarity_suspect"),
        "alpha_nontrivial": _flag_mask(proc_df, "alpha_nontrivial"),
        "no_foreground": _flag_mask(proc_df, "no_foreground"),
    }
    informational_summary = {
        name: {
            "count": int(mask.sum()),
            "pct_of_total": round(100.0 * mask.mean(), 2) if len(proc_df) else 0.0,
            "sample_image_ids": proc_df.loc[mask, "image_id"].head(20).tolist(),
        }
        for name, mask in informational.items()
    }

    critical_checks = {
        "output_count_matches_valid_input_count": len(proc_df) == len(valid_ids),
        "no_missing_outputs": len(missing_outputs) == 0,
        "no_unexpected_extra_metadata_rows": len(unexpected_extra_rows) == 0,
        "no_duplicate_image_ids": len(duplicate_image_ids) == 0,
        "no_missing_files_on_disk": len(missing_files_on_disk) == 0,
        "no_unreferenced_extra_files_on_disk": len(extra_files_on_disk) == 0,
        "all_ok_flagged_outputs_match_target_dimensions": len(dim_mismatch) == 0,
        "all_ok_flagged_outputs_in_valid_pixel_range": len(pixel_range_bad) == 0,
        "no_aspect_distortion_bug": len(aspect_distortion_bug) == 0,
    }

    critical_failures = [name for name, passed in critical_checks.items() if not passed]
    status = "FAIL" if critical_failures else "PASS"

    return {
        "status": status,
        "critical_checks": critical_checks,
        "critical_failures": critical_failures,
        "counts": {
            "phase1_valid_images": len(valid_ids),
            "preprocessing_metadata_rows": len(proc_df),
            "status_ok": int(len(ok_df)),
            "status_flagged": int(len(flagged_df)),
            "status_failed": int(len(failed_df)),
        },
        "missing_outputs_sample": missing_outputs[:20],
        "unexpected_extra_rows_sample": unexpected_extra_rows[:20],
        "duplicate_image_ids_sample": duplicate_image_ids[:20],
        "missing_files_on_disk_sample": missing_files_on_disk[:20],
        "extra_files_on_disk_sample": extra_files_on_disk[:20],
        "dimension_mismatch_sample": dim_mismatch["image_id"].head(20).tolist(),
        "pixel_range_bad_sample": pixel_range_bad["image_id"].head(20).tolist(),
        "aspect_distortion_bug_sample": aspect_distortion_bug["image_id"].head(20).tolist(),
        "failed_images_sample": failed_df[["image_id", "error"]].head(20).to_dict("records"),
        "informational_flags": informational_summary,
    }


def print_qc_summary(qc: dict) -> None:
    print("=" * 70)
    print("PHASE 2 / PART 6: FULL-BATCH QC")
    print("=" * 70)
    c = qc["counts"]
    print(f"Phase 1 valid images: {c['phase1_valid_images']}")
    print(f"Preprocessing metadata rows: {c['preprocessing_metadata_rows']}")
    print(f"  status=ok: {c['status_ok']}  status=flagged: {c['status_flagged']}  status=failed: {c['status_failed']}")

    print("\n--- CRITICAL checks ---")
    for name, passed in qc["critical_checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nOVERALL STATUS: {qc['status']}")
    if qc["critical_failures"]:
        print("Critical failures:")
        for cf in qc["critical_failures"]:
            print(f"  - {cf}")

    if qc["failed_images_sample"]:
        print(f"\nSample of FAILED images (file missing/corrupt, never silently dropped):")
        for item in qc["failed_images_sample"][:10]:
            print(f"  {item['image_id']}: {item['error']}")

    print("\n--- INFORMATIONAL flags (expected on real data; review, not failures) ---")
    for name, info in qc["informational_flags"].items():
        print(f"  {name}: {info['count']} images ({info['pct_of_total']}%)")
