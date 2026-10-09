"""
pairing/sampling.py

Phase 3B, Steps 5-6 -- turns the exhaustive candidate set
(pairing/candidates.py) into the final pair dataset by applying a
configurable, deterministic sampling strategy.

STRATEGY (default, see configs/pair_config.yaml):
  - ALL genuine_genuine (positive) pairs are kept. Genuine counts are
    exactly uniform (24 per writer) across every dataset per the real
    Phase 1 statistics, so no writer can dominate positive pairs, and
    "preserve all available positive pairs where practical" (Step 5,
    principle 1) is directly achievable here.
  - genuine_forgery (negative) pairs are capped PER WRITER at
    round(negative_to_positive_ratio * that writer's own positive pair
    count), not a global count. This is deliberately relative rather
    than a fixed global number: it directly satisfies "prevent writers
    with unusually many possible pairs from dominating training" (Step
    5, principle 2) regardless of how uniform or skewed the real
    per-writer distribution turns out to be, and it directly targets
    principle 3 ("maintain a reasonable positive/negative balance") via
    one explicit, configurable ratio (default 1.0 -- i.e. target ~1:1).
  - When a writer's possible negatives exceed its cap, a deterministic,
    seeded subsample is taken (Python's random.Random, seeded per-writer
    so each writer's sampling is independent and the overall result does
    not depend on writer processing order). When a writer's possible
    negatives are already at or below the cap, ALL of them are kept --
    sampling never "pads" negatives.
"""

from __future__ import annotations

import random

import pandas as pd


def sample_negative_pairs(candidate_pairs_df: pd.DataFrame, negative_to_positive_ratio: float,
                          seed: int) -> pd.DataFrame:
    """Applies the per-writer-relative negative cap described in this
    module's docstring. Positive pairs pass through unchanged. Returns a
    new DataFrame (does not mutate the input), with the same schema as
    candidate_pairs_df, re-sorted deterministically by pair_id.
    """
    positives = candidate_pairs_df[candidate_pairs_df["pair_type"] == "genuine_genuine"]
    negatives = candidate_pairs_df[candidate_pairs_df["pair_type"] == "genuine_forgery"]

    pos_counts = positives.groupby("global_writer_id").size().to_dict()

    kept_negative_rows = []
    for gw, group in negatives.groupby("global_writer_id"):
        cap = round(negative_to_positive_ratio * pos_counts.get(gw, 0))
        rows = group.sort_values("pair_id")  # deterministic base order
        if cap >= len(rows):
            kept_negative_rows.append(rows)
        else:
            rng = random.Random(f"{seed}:{gw}")
            idx = list(rows.index)
            chosen = rng.sample(idx, cap) if cap > 0 else []
            kept_negative_rows.append(rows.loc[sorted(chosen)])

    kept_negatives = pd.concat(kept_negative_rows, ignore_index=True) if kept_negative_rows else negatives.iloc[0:0]
    result = pd.concat([positives, kept_negatives], ignore_index=True)
    return result.sort_values("pair_id").reset_index(drop=True)


def summarize_sampling(candidate_pairs_df: pd.DataFrame, sampled_pairs_df: pd.DataFrame) -> dict:
    """Before/after counts, for the final report -- how many candidate
    negatives existed vs how many were actually kept, overall and per
    split/dataset.
    """
    def _counts(df: pd.DataFrame) -> dict:
        return {
            "positive": int((df["pair_type"] == "genuine_genuine").sum()),
            "negative": int((df["pair_type"] == "genuine_forgery").sum()),
        }

    before = _counts(candidate_pairs_df)
    after = _counts(sampled_pairs_df)

    by_split_before = candidate_pairs_df.groupby(["split", "pair_type"]).size().unstack(fill_value=0)
    by_split_after = sampled_pairs_df.groupby(["split", "pair_type"]).size().unstack(fill_value=0)

    return {
        "candidate_counts": before,
        "final_counts": after,
        "candidate_negative_to_positive_ratio": round(before["negative"] / before["positive"], 3) if before["positive"] else None,
        "final_negative_to_positive_ratio": round(after["negative"] / after["positive"], 3) if after["positive"] else None,
        "negatives_discarded": before["negative"] - after["negative"],
        "pct_negatives_discarded": round(100.0 * (before["negative"] - after["negative"]) / before["negative"], 2) if before["negative"] else 0.0,
        "by_split_before": by_split_before.to_dict("index"),
        "by_split_after": by_split_after.to_dict("index"),
    }
