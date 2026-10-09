"""
preprocessing/duplicate_audit.py

Phase 2, Part 2 -- audits the duplicate groups Phase 1 already found
(duplicate_report.csv) against the real metadata (metadata.csv), to
determine WHAT KIND of duplication each group represents before any
preprocessing or splitting decision is made.

This module NEVER deletes or modifies anything. It only reads the two
existing Phase 1 CSVs and produces a classified report.

For every duplicate group (one row per hash_type+hash_value in
duplicate_report.csv), this computes:
  - how many images are in the group
  - how many distinct datasets / global_writer_ids are involved
  - whether the group is confined to one writer, one dataset
  - the multiset of signature_type values present (genuine/forgery/mixed)
  - a leakage_risk flag: True when the SAME real signature (an exact
    sha256 duplicate) is shared by more than one global_writer_id --
    that is the specific pattern that would let the same physical image
    appear on both sides of a future writer-disjoint split.

Perceptual (phash) cross-writer/cross-dataset groups are reported but
NOT flagged as leakage_risk by default, because (as observed on this
project's real Bengali/Hindi data) phash on mostly-blank, thin-stroke
signature scans produces a meaningful rate of coincidental collisions
between genuinely different signatures -- treating every phash group as
a confirmed duplicate would overstate the problem. They are surfaced
separately (`perceptual_cross_writer_groups`) so a human can spot-check
representative examples rather than trust the hash blindly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_duplicate_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_ids"] = df["image_ids"].apply(
        lambda s: [x for x in str(s).split(";") if x]
    )
    return df


def _build_image_lookup(metadata_df: pd.DataFrame) -> dict:
    """image_id -> {dataset, global_writer_id, signature_type}"""
    cols = ["image_id", "dataset", "global_writer_id", "signature_type", "writer_id"]
    lookup = {}
    for row in metadata_df[cols].itertuples(index=False):
        lookup[row.image_id] = {
            "dataset": row.dataset,
            "global_writer_id": row.global_writer_id,
            "signature_type": row.signature_type,
            "writer_id": row.writer_id,
        }
    return lookup


def classify_groups(dup_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a long-format DataFrame: one row per (group, image_id),
    annotated with that image's provenance and the group's overall
    classification -- this is what gets written to
    reports/phase2_duplicate_audit.csv.
    """
    lookup = _build_image_lookup(metadata_df)
    rows = []

    for grp in dup_df.itertuples(index=False):
        image_ids = grp.image_ids
        infos = [lookup.get(iid) for iid in image_ids]
        missing = [iid for iid, info in zip(image_ids, infos) if info is None]
        present = [(iid, info) for iid, info in zip(image_ids, infos) if info is not None]

        datasets = {info["dataset"] for _, info in present}
        writers = {info["global_writer_id"] for _, info in present}
        sig_types = {info["signature_type"] for _, info in present}

        same_dataset = len(datasets) == 1
        same_writer = len(writers) == 1

        if sig_types == {"genuine"}:
            label_composition = "genuine_genuine"
        elif sig_types == {"forgery"}:
            label_composition = "forgery_forgery"
        elif sig_types == {"genuine", "forgery"}:
            label_composition = "genuine_forgery_mixed"
        else:
            label_composition = "other_or_unknown"

        # Leakage risk: the SAME physical file (exact hash) shared across
        # more than one writer identity is the pattern that matters for
        # future writer-disjoint splitting.
        leakage_risk = (grp.hash_type == "sha256_exact") and (not same_writer)

        for iid, info in present:
            rows.append({
                "hash_type": grp.hash_type,
                "hash_value": grp.hash_value,
                "group_size": len(image_ids),
                "group_missing_image_ids": len(missing),
                "image_id": iid,
                "dataset": info["dataset"],
                "writer_id": info["writer_id"],
                "global_writer_id": info["global_writer_id"],
                "signature_type": info["signature_type"],
                "group_same_dataset": same_dataset,
                "group_same_writer": same_writer,
                "group_label_composition": label_composition,
                "group_leakage_risk": leakage_risk,
            })
        for iid in missing:
            rows.append({
                "hash_type": grp.hash_type,
                "hash_value": grp.hash_value,
                "group_size": len(image_ids),
                "group_missing_image_ids": len(missing),
                "image_id": iid,
                "dataset": None,
                "writer_id": None,
                "global_writer_id": None,
                "signature_type": None,
                "group_same_dataset": None,
                "group_same_writer": None,
                "group_label_composition": "MISSING_FROM_METADATA",
                "group_leakage_risk": None,
            })

    return pd.DataFrame(rows)


def summarize_audit(audit_df: pd.DataFrame) -> dict:
    """Produces the aggregate answers to Part 2's questions A-C, F."""

    def _n_images_in(hash_type: str) -> int:
        sub = audit_df[audit_df["hash_type"] == hash_type]
        return int(sub["image_id"].nunique())

    def _group_level(hash_type: str) -> pd.DataFrame:
        sub = audit_df[audit_df["hash_type"] == hash_type]
        return sub.drop_duplicates(subset=["hash_type", "hash_value"])

    exact_groups = _group_level("sha256_exact")
    perceptual_groups = _group_level("phash_perceptual")

    def _group_counts(groups: pd.DataFrame) -> dict:
        if groups.empty:
            return {
                "num_groups": 0, "same_writer_groups": 0, "cross_writer_groups": 0,
                "same_dataset_groups": 0, "cross_dataset_groups": 0,
                "genuine_genuine_groups": 0, "forgery_forgery_groups": 0,
                "genuine_forgery_mixed_groups": 0,
            }
        return {
            "num_groups": int(len(groups)),
            "same_writer_groups": int((groups["group_same_writer"] == True).sum()),  # noqa: E712
            "cross_writer_groups": int((groups["group_same_writer"] == False).sum()),  # noqa: E712
            "same_dataset_groups": int((groups["group_same_dataset"] == True).sum()),  # noqa: E712
            "cross_dataset_groups": int((groups["group_same_dataset"] == False).sum()),  # noqa: E712
            "genuine_genuine_groups": int((groups["group_label_composition"] == "genuine_genuine").sum()),
            "forgery_forgery_groups": int((groups["group_label_composition"] == "forgery_forgery").sum()),
            "genuine_forgery_mixed_groups": int((groups["group_label_composition"] == "genuine_forgery_mixed").sum()),
        }

    leakage_rows = audit_df[audit_df["group_leakage_risk"] == True]  # noqa: E712
    leakage_groups = leakage_rows.drop_duplicates(subset=["hash_type", "hash_value"])

    summary = {
        "A_images_in_exact_duplicate_groups": _n_images_in("sha256_exact"),
        "B_images_in_perceptual_duplicate_groups": _n_images_in("phash_perceptual"),
        "C_exact_duplicate_group_breakdown": _group_counts(exact_groups),
        "C_perceptual_duplicate_group_breakdown": _group_counts(perceptual_groups),
        "F_leakage_risk": {
            "definition": (
                "An EXACT (sha256) duplicate group spanning more than one "
                "global_writer_id -- i.e. the identical physical image file "
                "is attributed to more than one writer identity. This is the "
                "pattern that could leak the same image across both sides of "
                "a future writer-disjoint train/test split."
            ),
            "num_flagged_groups": int(len(leakage_groups)),
            "num_flagged_images": int(leakage_rows["image_id"].nunique()),
            "flagged_group_hash_values": leakage_groups["hash_value"].tolist()[:50],
        },
        "perceptual_cross_writer_note": (
            "Perceptual (phash) cross-writer/cross-dataset groups are common "
            "on this project's real data and are NOT automatically counted as "
            "leakage risk -- phash collisions on mostly-blank, thin-stroke "
            "signature scans include a meaningful rate of false positives "
            "(visually different signatures hashing to the same bucket). "
            "See group_label_composition / group_same_writer columns in the "
            "full audit CSV to spot-check representative phash groups "
            "yourself before deciding whether any specific one is real."
        ),
        "missing_from_metadata": int(
            (audit_df["group_label_composition"] == "MISSING_FROM_METADATA").sum()
        ),
    }
    return summary


def print_audit_summary(summary: dict) -> None:
    print("=" * 60)
    print("PHASE 2 / PART 2: DUPLICATE AUDIT SUMMARY")
    print("=" * 60)
    print(f"A. Images in EXACT duplicate groups:       {summary['A_images_in_exact_duplicate_groups']}")
    print(f"B. Images in PERCEPTUAL duplicate groups:  {summary['B_images_in_perceptual_duplicate_groups']}")

    print("\nC. Exact-duplicate group breakdown:")
    for k, v in summary["C_exact_duplicate_group_breakdown"].items():
        print(f"    {k}: {v}")

    print("\nC. Perceptual-duplicate group breakdown:")
    for k, v in summary["C_perceptual_duplicate_group_breakdown"].items():
        print(f"    {k}: {v}")

    fl = summary["F_leakage_risk"]
    print(f"\nF. LEAKAGE RISK -- {fl['num_flagged_groups']} exact-duplicate groups "
          f"span multiple writer identities ({fl['num_flagged_images']} images total).")
    if fl["num_flagged_groups"] > 0:
        print("   *** This needs manual review before writer-disjoint splitting "
              "in a later phase -- see the flagged hash values in the JSON report. ***")
    else:
        print("   None found: every exact-duplicate group stays within a single writer.")

    print(f"\n{summary['perceptual_cross_writer_note']}")

    if summary["missing_from_metadata"]:
        print(f"\nWARNING: {summary['missing_from_metadata']} image_ids referenced in "
              f"duplicate_report.csv were NOT found in metadata.csv -- investigate before proceeding.")
