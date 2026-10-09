"""
pairing/qc.py

Phase 3B, Step 13 -- the full pair-level QC checklist, run against the
FINAL pairs (after sampling). Every check from the spec's checklist is
implemented as its own function so failures are individually attributable,
then combined into one report.
"""

from __future__ import annotations

import pandas as pd


def check_no_split_crossing(pairs_df: pd.DataFrame, image_split_map: pd.DataFrame) -> dict:
    lookup = image_split_map.set_index("image_id")["split"].to_dict()
    img1_split = pairs_df["image_1_id"].map(lookup)
    img2_split = pairs_df["image_2_id"].map(lookup)
    mismatched = pairs_df[(img1_split != img2_split) | (img1_split != pairs_df["split"])]
    return {"passed": len(mismatched) == 0, "num_violations": int(len(mismatched)),
            "sample_violations": mismatched["pair_id"].head(10).tolist()}


def check_writer_consistency(pairs_df: pd.DataFrame, image_metadata_df: pd.DataFrame) -> dict:
    lookup = image_metadata_df.set_index("image_id")["global_writer_id"].to_dict()
    img1_writer = pairs_df["image_1_id"].map(lookup)
    img2_writer = pairs_df["image_2_id"].map(lookup)
    mismatched = pairs_df[(img1_writer != img2_writer) | (img1_writer != pairs_df["global_writer_id"])]
    return {"passed": len(mismatched) == 0, "num_violations": int(len(mismatched)),
            "sample_violations": mismatched["pair_id"].head(10).tolist()}


def check_label_type_consistency(pairs_df: pd.DataFrame, image_metadata_df: pd.DataFrame) -> dict:
    lookup = image_metadata_df.set_index("image_id")["signature_type"].to_dict()
    img1_type = pairs_df["image_1_id"].map(lookup)
    img2_type = pairs_df["image_2_id"].map(lookup)

    is_gg = pairs_df["pair_type"] == "genuine_genuine"
    is_gf = pairs_df["pair_type"] == "genuine_forgery"

    gg_bad = pairs_df[is_gg & ~((img1_type == "genuine") & (img2_type == "genuine"))]
    gf_bad = pairs_df[is_gf & ~(
        ((img1_type == "genuine") & (img2_type == "forgery"))
    )]
    label_bad = pairs_df[((pairs_df["pair_type"] == "genuine_genuine") & (pairs_df["pair_label"] != 1))
                          | ((pairs_df["pair_type"] == "genuine_forgery") & (pairs_df["pair_label"] != 0))]

    violations = pd.concat([gg_bad, gf_bad, label_bad]).drop_duplicates(subset="pair_id")
    return {"passed": len(violations) == 0, "num_violations": int(len(violations)),
            "sample_violations": violations["pair_id"].head(10).tolist()}


def check_no_ineligible_writer_pairs(pairs_df: pd.DataFrame, manifest_df: pd.DataFrame) -> dict:
    eligible_writers = set(manifest_df.loc[manifest_df["pair_generation_eligible"] == True, "global_writer_id"])  # noqa: E712
    violating = pairs_df[~pairs_df["global_writer_id"].isin(eligible_writers)]
    return {"passed": len(violating) == 0, "num_violations": int(len(violating)),
            "violating_writers": sorted(violating["global_writer_id"].unique().tolist())}


def check_no_duplicate_pair_ids(pairs_df: pd.DataFrame) -> dict:
    dup = pairs_df["pair_id"][pairs_df["pair_id"].duplicated()]
    return {"passed": len(dup) == 0, "num_duplicates": int(len(dup)), "sample": dup.head(10).tolist()}


def check_no_reversed_duplicate_pairs(pairs_df: pd.DataFrame) -> dict:
    """A reversed duplicate would be two rows with the same (pair_type,
    global_writer_id) and {image_1_id, image_2_id} as an unordered set,
    but swapped image_1/image_2. Canonical construction should make this
    impossible by design -- this check verifies that held.
    """
    key = pairs_df.apply(
        lambda r: (r["pair_type"], r["global_writer_id"], frozenset((r["image_1_id"], r["image_2_id"]))),
        axis=1,
    )
    dup = key[key.duplicated()]
    return {"passed": len(dup) == 0, "num_violations": int(len(dup))}


def check_images_exist_and_match_metadata(pairs_df: pd.DataFrame, image_metadata_df: pd.DataFrame) -> dict:
    known_ids = set(image_metadata_df["image_id"])
    img1_missing = ~pairs_df["image_1_id"].isin(known_ids)
    img2_missing = ~pairs_df["image_2_id"].isin(known_ids)
    missing = pairs_df[img1_missing | img2_missing]
    return {"passed": len(missing) == 0, "num_missing_reference_rows": int(len(missing)),
            "sample": missing["pair_id"].head(10).tolist()}


def check_bhsig_hindi_123_produces_zero_pairs(pairs_df: pd.DataFrame, writer_id: str = "BHSIG_HINDI_123") -> dict:
    count = int((pairs_df["global_writer_id"] == writer_id).sum())
    return {"passed": count == 0, "writer_id": writer_id, "pairs_found": count}


def check_split_assignments_unchanged(manifest_df: pd.DataFrame, original_manifest_df: pd.DataFrame) -> dict:
    """Verifies Phase 3A's split assignment was not altered anywhere in
    this phase -- compares the manifest used here against a reference
    copy loaded fresh from disk.
    """
    merged = manifest_df.merge(original_manifest_df, on="global_writer_id", suffixes=("_used", "_original"))
    changed = merged[merged["split_used"] != merged["split_original"]]
    return {"passed": len(changed) == 0, "num_changed": int(len(changed)),
            "changed_writers": changed["global_writer_id"].tolist()}


def check_perceptual_cross_split_pair_overlap(pairs_df: pd.DataFrame, dup_audit_df: pd.DataFrame | None,
                                              image_split_map: pd.DataFrame) -> dict:
    """INFORMATIONAL (Step 9): reports whether any generated pair contains
    an image that belongs to a phash group spanning multiple splits. This
    is reported, never auto-failed -- consistent with Phase 2/3A's
    established position that phash collisions include false positives.
    """
    if dup_audit_df is None or len(dup_audit_df) == 0:
        return {"note": "no duplicate audit data provided", "num_pairs_involved": 0}

    image_to_split = image_split_map.set_index("image_id")["split"].to_dict()
    perceptual = dup_audit_df[dup_audit_df["hash_type"] == "phash_perceptual"]

    spanning_images = set()
    for _, group in perceptual.groupby("hash_value"):
        ids = group["image_id"].dropna().tolist()
        splits = {image_to_split.get(i) for i in ids}
        splits.discard(None)
        if len(splits) > 1:
            spanning_images.update(ids)

    involved = pairs_df[pairs_df["image_1_id"].isin(spanning_images) | pairs_df["image_2_id"].isin(spanning_images)]
    return {
        "num_pairs_involved": int(len(involved)),
        "pct_of_total_pairs": round(100.0 * len(involved) / len(pairs_df), 3) if len(pairs_df) else 0.0,
        "sample_pair_ids": involved["pair_id"].head(10).tolist(),
        "note": "informational only -- not treated as leakage, consistent with Phase 2/3A findings",
    }


def run_full_qc(*, pairs_df: pd.DataFrame, manifest_df: pd.DataFrame, image_metadata_df: pd.DataFrame,
                image_split_map: pd.DataFrame, dup_audit_df: pd.DataFrame | None,
                original_manifest_df: pd.DataFrame) -> dict:
    critical_checks = {
        "no_pair_crosses_split": check_no_split_crossing(pairs_df, image_split_map),
        "no_pair_contains_different_writers": check_writer_consistency(pairs_df, image_metadata_df),
        "pair_labels_match_pair_types": check_label_type_consistency(pairs_df, image_metadata_df),
        "no_ineligible_writer_produces_pairs": check_no_ineligible_writer_pairs(pairs_df, manifest_df),
        "no_duplicate_pair_ids": check_no_duplicate_pair_ids(pairs_df),
        "no_reversed_duplicate_pairs": check_no_reversed_duplicate_pairs(pairs_df),
        "every_referenced_image_exists": check_images_exist_and_match_metadata(pairs_df, image_metadata_df),
        "bhsig_hindi_123_produces_zero_pairs": check_bhsig_hindi_123_produces_zero_pairs(pairs_df),
        "no_phase3a_split_assignments_changed": check_split_assignments_unchanged(manifest_df, original_manifest_df),
    }

    critical_failures = [name for name, result in critical_checks.items() if not result["passed"]]
    status = "FAIL" if critical_failures else "PASS"

    perceptual_overlap = check_perceptual_cross_split_pair_overlap(pairs_df, dup_audit_df, image_split_map)

    pair_counts_by_writer = pairs_df.groupby("global_writer_id").size()
    distribution_report = {
        "total_pairs": int(len(pairs_df)),
        "by_split": pairs_df.groupby("split").size().to_dict(),
        "by_dataset": pairs_df.groupby("dataset").size().to_dict(),
        "by_pair_type": pairs_df.groupby("pair_type").size().to_dict(),
        "by_pair_label": pairs_df.groupby("pair_label").size().to_dict(),
        "per_writer_pair_count_stats": pair_counts_by_writer.describe().to_dict() if len(pair_counts_by_writer) else {},
        "writers_with_unusually_low_count": pair_counts_by_writer[
            pair_counts_by_writer < pair_counts_by_writer.mean() - 2 * pair_counts_by_writer.std()
        ].to_dict() if len(pair_counts_by_writer) > 1 else {},
        "writers_with_unusually_high_count": pair_counts_by_writer[
            pair_counts_by_writer > pair_counts_by_writer.mean() + 2 * pair_counts_by_writer.std()
        ].to_dict() if len(pair_counts_by_writer) > 1 else {},
    }

    return {
        "status": status,
        "critical_checks": {name: result["passed"] for name, result in critical_checks.items()},
        "critical_check_details": critical_checks,
        "critical_failures": critical_failures,
        "perceptual_cross_split_pair_overlap": perceptual_overlap,
        "distribution_report": distribution_report,
    }


def print_qc_summary(qc: dict) -> None:
    print("=" * 70)
    print("PHASE 3B / STEP 13: PAIR-LEVEL QC")
    print("=" * 70)
    for name, passed in qc["critical_checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nOVERALL STATUS: {qc['status']}")
    if qc["critical_failures"]:
        for cf in qc["critical_failures"]:
            detail = qc["critical_check_details"][cf]
            print(f"  FAILURE DETAIL ({cf}): {detail}")

    po = qc["perceptual_cross_split_pair_overlap"]
    print(f"\nPerceptual cross-split pair overlap (informational): "
          f"{po.get('num_pairs_involved', 0)} pairs ({po.get('pct_of_total_pairs', 0)}%)")

    dr = qc["distribution_report"]
    print(f"\nTotal final pairs: {dr['total_pairs']:,}")
    print(f"By split: {dr['by_split']}")
    print(f"By dataset: {dr['by_dataset']}")
    print(f"By pair_type: {dr['by_pair_type']}")
    print(f"By pair_label: {dr['by_pair_label']}")
    if dr["writers_with_unusually_low_count"]:
        print(f"Writers with unusually LOW pair counts: {dr['writers_with_unusually_low_count']}")
    if dr["writers_with_unusually_high_count"]:
        print(f"Writers with unusually HIGH pair counts: {dr['writers_with_unusually_high_count']}")
