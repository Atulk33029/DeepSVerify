"""
splitting/split_audit.py

Phase 3A audit: verifies the split assignment from writer_split.py is
actually sound before it's trusted for Phase 3B pair generation.

CRITICAL checks (any failure means the split itself is broken):
  - every writer assigned to exactly one split (no duplicates, none missing)
  - every split is non-empty and every dataset is represented in every split
  - no EXACT (sha256) duplicate group spans more than one split -- this
    should be structurally impossible (Phase 2's duplicate audit already
    found all exact-duplicate groups are confined to a single writer, and
    a writer can only be in one split), so this check mainly verifies
    that guarantee held end-to-end rather than discovering a new problem

INFORMATIONAL (reported, never auto-failed):
  - PERCEPTUAL (phash) duplicate groups that span multiple splits. These
    are NOT automatically treated as leakage, per the Phase 2 finding
    that phash collisions on thin-stroke signature scans include a
    meaningful false-positive rate. Reported with counts/examples for
    human review, same posture as Phase 2's duplicate audit.
  - writers with insufficient genuine/forgery counts to support pair
    generation (flagged, not excluded).
"""

from __future__ import annotations

import pandas as pd


def check_writer_exclusivity(writer_split_df: pd.DataFrame) -> dict:
    counts = writer_split_df["global_writer_id"].value_counts()
    duplicated = counts[counts > 1].index.tolist()
    return {
        "passed": len(duplicated) == 0,
        "duplicated_writers": duplicated,
        "total_writers": int(len(writer_split_df)),
        "total_distinct_writers": int(writer_split_df["global_writer_id"].nunique()),
    }


def check_no_missing_writers(original_writer_ids: set, writer_split_df: pd.DataFrame) -> dict:
    assigned = set(writer_split_df["global_writer_id"])
    missing = sorted(original_writer_ids - assigned)
    unexpected = sorted(assigned - original_writer_ids)
    return {
        "passed": len(missing) == 0 and len(unexpected) == 0,
        "missing_writers": missing,
        "unexpected_writers": unexpected,
    }


def check_dataset_representation(writer_split_df: pd.DataFrame) -> dict:
    table = writer_split_df.groupby(["split", "dataset"]).size().unstack(fill_value=0)
    all_datasets = sorted(writer_split_df["dataset"].unique())
    all_splits = sorted(writer_split_df["split"].unique())
    zero_cells = [
        (split, dataset) for split in all_splits for dataset in all_datasets
        if table.loc[split, dataset] == 0
    ] if not table.empty else []
    return {
        "passed": len(zero_cells) == 0,
        "writer_counts_table": table.to_dict(),
        "zero_representation_cells": zero_cells,
    }


def check_image_counts_by_split(image_split_map: pd.DataFrame) -> dict:
    table = image_split_map.groupby(["split", "dataset"]).size().unstack(fill_value=0)
    totals = image_split_map.groupby("split").size().to_dict()
    label_table = image_split_map.groupby(["split", "signature_type"]).size().unstack(fill_value=0)
    return {
        "images_by_split_and_dataset": table.to_dict(),
        "total_images_by_split": {k: int(v) for k, v in totals.items()},
        "images_by_split_and_label": label_table.to_dict(),
    }


def check_class_availability(writer_split_df: pd.DataFrame, min_genuine: int, min_forgery: int) -> dict:
    """Reports class availability. `passed` here is informational, not a
    critical build_leakage_report failure (see that function): writers
    with insufficient genuine/forgery counts are a known, reported, real
    data characteristic (confirmed via raw-file inspection for
    BHSIG_HINDI_123 -- source folder genuinely contains zero genuine
    files), not something the split itself can or should fix. Every
    insufficient writer is named explicitly here, in full, never just
    summarized as a count -- this list is small enough on real data
    (1 of 315 writers) that there is no reason to truncate or bury it.
    """
    insufficient = writer_split_df[
        (writer_split_df["n_genuine"] < min_genuine) | (writer_split_df["n_forgery"] < min_forgery)
    ]
    by_split = insufficient.groupby("split").size().to_dict() if len(insufficient) else {}

    named = []
    for row in insufficient.itertuples(index=False):
        reason = (
            "source dataset contains no genuine files for this writer" if row.n_genuine == 0
            else (
                f"insufficient genuine signatures (n_genuine={row.n_genuine}, need >={min_genuine})"
                if row.n_genuine < min_genuine
                else f"insufficient forgery signatures (n_forgery={row.n_forgery}, need >={min_forgery})"
            )
        )
        named.append({
            "global_writer_id": row.global_writer_id,
            "dataset": row.dataset,
            "split": row.split,
            "n_genuine": int(row.n_genuine),
            "n_forgery": int(row.n_forgery),
            "reason": reason,
            "pair_generation_eligible": False,
        })

    return {
        "passed": len(insufficient) == 0,
        "min_genuine_required": min_genuine,
        "min_forgery_required": min_forgery,
        "num_writers_insufficient": int(len(insufficient)),
        "insufficient_writers_by_split": by_split,
        "insufficient_writers_named": named,
    }


def check_duplicate_group_cross_split(dup_audit_df: pd.DataFrame, image_split_map: pd.DataFrame,
                                      hash_type: str) -> dict:
    """For every duplicate group of the given hash_type, checks whether
    its member images landed in more than one split. Returns per-group
    detail plus an aggregate summary. This function itself never decides
    pass/fail -- the caller interprets exact vs perceptual differently
    (see module docstring).
    """
    image_to_split = image_split_map.set_index("image_id")["split"].to_dict()
    sub = dup_audit_df[dup_audit_df["hash_type"] == hash_type]

    groups = []
    for hash_value, group in sub.groupby("hash_value"):
        image_ids = group["image_id"].dropna().tolist()
        splits = {image_to_split.get(iid) for iid in image_ids}
        splits.discard(None)
        groups.append({
            "hash_value": hash_value,
            "num_images": len(image_ids),
            "splits_involved": sorted(splits),
            "spans_multiple_splits": len(splits) > 1,
        })

    spanning = [g for g in groups if g["spans_multiple_splits"]]
    return {
        "hash_type": hash_type,
        "total_groups": len(groups),
        "groups_spanning_multiple_splits": len(spanning),
        "pct_spanning": round(100.0 * len(spanning) / len(groups), 2) if groups else 0.0,
        "spanning_examples": spanning[:20],
    }


def build_leakage_report(*, writer_split_df: pd.DataFrame, original_writer_ids: set,
                         image_split_map: pd.DataFrame, dup_audit_df: pd.DataFrame | None,
                         min_genuine: int, min_forgery: int) -> dict:
    exclusivity = check_writer_exclusivity(writer_split_df)
    no_missing = check_no_missing_writers(original_writer_ids, writer_split_df)
    representation = check_dataset_representation(writer_split_df)
    image_counts = check_image_counts_by_split(image_split_map)
    class_availability = check_class_availability(writer_split_df, min_genuine, min_forgery)

    exact_result = perceptual_result = None
    exact_critical_pass = True
    if dup_audit_df is not None and len(dup_audit_df):
        exact_result = check_duplicate_group_cross_split(dup_audit_df, image_split_map, "sha256_exact")
        perceptual_result = check_duplicate_group_cross_split(dup_audit_df, image_split_map, "phash_perceptual")
        exact_critical_pass = exact_result["groups_spanning_multiple_splits"] == 0

    critical_checks = {
        "writer_exclusivity": exclusivity["passed"],
        "no_missing_or_unexpected_writers": no_missing["passed"],
        "every_split_has_every_dataset_represented": representation["passed"],
        "no_exact_duplicate_group_spans_multiple_splits": exact_critical_pass,
    }
    critical_failures = [k for k, v in critical_checks.items() if not v]
    status = "FAIL" if critical_failures else "PASS"

    return {
        "status": status,
        "critical_checks": critical_checks,
        "critical_failures": critical_failures,
        "writer_exclusivity": exclusivity,
        "no_missing_writers": no_missing,
        "dataset_representation": representation,
        "image_counts": image_counts,
        "class_availability": class_availability,
        # Top-level, impossible-to-miss restatement of every
        # pair-generation-ineligible writer -- not buried in a count.
        # Deliberately duplicated from class_availability's own named
        # list (same source, same mechanical derivation) so this
        # specific finding is visible without digging into a nested key.
        "known_limitations": class_availability["insufficient_writers_named"],
        "exact_duplicate_cross_split": exact_result,
        "perceptual_duplicate_cross_split": perceptual_result,
        "perceptual_duplicate_note": (
            "Perceptual (phash) duplicate groups spanning multiple splits are reported above "
            "but NOT treated as a critical failure -- consistent with the Phase 2 finding that "
            "phash collisions on thin-stroke signature scans include a meaningful false-positive "
            "rate. Review spanning_examples manually if the spanning rate looks high; do not "
            "assume every spanning group is real leakage." if perceptual_result else None
        ),
    }


def print_leakage_report(report: dict) -> None:
    print("=" * 70)
    print("PHASE 3A: SPLIT LEAKAGE / QC REPORT")
    print("=" * 70)
    for name, passed in report["critical_checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nOVERALL STATUS: {report['status']}")
    if report["critical_failures"]:
        for cf in report["critical_failures"]:
            print(f"  FAILURE: {cf}")

    print(f"\nWriters: {report['writer_exclusivity']['total_distinct_writers']} distinct, "
          f"{report['writer_exclusivity']['total_writers']} total rows")

    print("\nWriter counts by split x dataset:")
    for split, row in report["dataset_representation"]["writer_counts_table"].items():
        print(f"  {split}: {row}")

    print("\nImage counts by split:")
    for split, n in report["image_counts"]["total_images_by_split"].items():
        print(f"  {split}: {n}")
    print("Image counts by split x signature_type:")
    for label, row in report["image_counts"]["images_by_split_and_label"].items():
        print(f"  {label}: {row}")

    ca = report["class_availability"]
    print(f"\nClass availability (min_genuine={ca['min_genuine_required']}, "
          f"min_forgery={ca['min_forgery_required']}): "
          f"{ca['num_writers_insufficient']} writers insufficient")
    if ca["num_writers_insufficient"]:
        print(f"  by split: {ca['insufficient_writers_by_split']}")
        print("\n  KNOWN LIMITATIONS (named explicitly, pair_generation_eligible=False):")
        for w in report["known_limitations"]:
            print(f"    - {w['global_writer_id']} ({w['dataset']}, split={w['split']}): "
                  f"n_genuine={w['n_genuine']}, n_forgery={w['n_forgery']} -- {w['reason']}")

    if report["exact_duplicate_cross_split"]:
        ex = report["exact_duplicate_cross_split"]
        print(f"\nExact duplicate groups: {ex['total_groups']} total, "
              f"{ex['groups_spanning_multiple_splits']} span multiple splits ({ex['pct_spanning']}%)")
    if report["perceptual_duplicate_cross_split"]:
        pc = report["perceptual_duplicate_cross_split"]
        print(f"Perceptual duplicate groups: {pc['total_groups']} total, "
              f"{pc['groups_spanning_multiple_splits']} span multiple splits ({pc['pct_spanning']}%) "
              f"-- informational, not auto-failed")
        if report["perceptual_duplicate_note"]:
            print(f"  {report['perceptual_duplicate_note']}")
