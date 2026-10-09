"""
pairing/measurement.py

Phase 3B, Steps 1 & 5 (measurement half) -- computes the THEORETICAL
number of possible pairs per writer, before any generation or sampling
happens:

    possible_positive (genuine_genuine) = C(n_genuine, 2)
    possible_negative (genuine_forgery) = n_genuine * n_forgery

Reads these counts directly from the Phase 3A split manifest
(n_genuine, n_forgery, pair_generation_eligible) -- no image loading
required, so this is cheap enough to run before committing to any
sampling strategy.

Ineligible writers (pair_generation_eligible == False, e.g.
BHSIG_HINDI_123) always get 0/0 here -- measurement must agree with the
eligibility rule used by actual generation, or the two could silently
diverge.
"""

from __future__ import annotations

from math import comb

import pandas as pd


def compute_theoretical_pair_counts(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Returns manifest_df with two added columns: possible_positive_pairs,
    possible_negative_pairs. Ineligible writers get 0 for both.
    """
    df = manifest_df.copy()
    eligible = df["pair_generation_eligible"].astype(bool)
    n_genuine = df["n_genuine"].astype(int)
    n_forgery = df["n_forgery"].astype(int)

    df["possible_positive_pairs"] = [
        comb(int(g), 2) if elig else 0 for elig, g in zip(eligible, n_genuine)
    ]
    df["possible_negative_pairs"] = [
        int(g) * int(f) if elig else 0 for elig, g, f in zip(eligible, n_genuine, n_forgery)
    ]
    return df


def aggregate_pair_counts(counts_df: pd.DataFrame) -> dict:
    """Aggregates the per-writer theoretical counts into the full
    reporting breakdown Step 1/5/10 ask for: totals, by-split, by-dataset,
    by-split-and-dataset, and per-writer distribution statistics
    (restricted to eligible writers, since ineligible ones are
    structurally 0/0 and would just distort a 'distribution').
    """
    eligible = counts_df[counts_df["pair_generation_eligible"] == True]  # noqa: E712
    ineligible = counts_df[counts_df["pair_generation_eligible"] == False]  # noqa: E712

    total_pos = int(counts_df["possible_positive_pairs"].sum())
    total_neg = int(counts_df["possible_negative_pairs"].sum())

    by_split = counts_df.groupby("split")[["possible_positive_pairs", "possible_negative_pairs"]].sum()
    by_dataset = counts_df.groupby("dataset")[["possible_positive_pairs", "possible_negative_pairs"]].sum()
    by_split_dataset = counts_df.groupby(["split", "dataset"])[
        ["possible_positive_pairs", "possible_negative_pairs"]
    ].sum()

    pos_stats = eligible["possible_positive_pairs"].describe().to_dict() if len(eligible) else {}
    neg_stats = eligible["possible_negative_pairs"].describe().to_dict() if len(eligible) else {}

    return {
        "num_total_writers": int(len(counts_df)),
        "num_eligible_writers": int(len(eligible)),
        "num_ineligible_writers": int(len(ineligible)),
        "ineligible_writer_ids": sorted(ineligible["global_writer_id"].tolist()),
        "total_possible_positive_pairs": total_pos,
        "total_possible_negative_pairs": total_neg,
        "negative_to_positive_ratio_overall": round(total_neg / total_pos, 3) if total_pos else None,
        "by_split": by_split.to_dict("index"),
        "by_dataset": by_dataset.to_dict("index"),
        "by_split_and_dataset": {str(k): v for k, v in by_split_dataset.to_dict("index").items()},
        "per_writer_positive_pair_stats": pos_stats,
        "per_writer_negative_pair_stats": neg_stats,
    }


def print_measurement_summary(agg: dict) -> None:
    print("=" * 70)
    print("PHASE 3B / STEP 1 & 5: THEORETICAL PAIR COUNT MEASUREMENT")
    print("=" * 70)
    print(f"Total writers: {agg['num_total_writers']}  "
          f"(eligible: {agg['num_eligible_writers']}, ineligible: {agg['num_ineligible_writers']})")
    if agg["ineligible_writer_ids"]:
        print(f"Ineligible writer(s): {agg['ineligible_writer_ids']}")

    print(f"\nTotal POSSIBLE positive (genuine_genuine) pairs: {agg['total_possible_positive_pairs']:,}")
    print(f"Total POSSIBLE negative (genuine_forgery) pairs: {agg['total_possible_negative_pairs']:,}")
    print(f"Overall negative:positive ratio: {agg['negative_to_positive_ratio_overall']}")

    print("\nBy split:")
    for split, row in agg["by_split"].items():
        print(f"  {split}: positive={row['possible_positive_pairs']:,}  negative={row['possible_negative_pairs']:,}")

    print("\nBy dataset:")
    for dataset, row in agg["by_dataset"].items():
        print(f"  {dataset}: positive={row['possible_positive_pairs']:,}  negative={row['possible_negative_pairs']:,}")

    print("\nBy split x dataset:")
    for key, row in agg["by_split_and_dataset"].items():
        print(f"  {key}: positive={row['possible_positive_pairs']:,}  negative={row['possible_negative_pairs']:,}")

    if agg["per_writer_positive_pair_stats"]:
        p, n = agg["per_writer_positive_pair_stats"], agg["per_writer_negative_pair_stats"]
        print(f"\nPer-writer positive pairs (eligible writers): "
              f"mean={p.get('mean', 0):.1f} min={p.get('min', 0):.0f} max={p.get('max', 0):.0f}")
        print(f"Per-writer negative pairs (eligible writers): "
              f"mean={n.get('mean', 0):.1f} min={n.get('min', 0):.0f} max={n.get('max', 0):.0f}")
