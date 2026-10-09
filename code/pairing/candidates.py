"""
pairing/candidates.py

Phase 3B, Steps 2-4, 7, 8 -- exhaustive, deterministic candidate pair
construction. "Exhaustive" here means every valid pair is produced before
any subsampling (pairing/sampling.py is a separate, later step) -- this
module's output already satisfies every structural invariant on its own:

  - split/writer isolation is structural, not just checked: pairs are
    built one writer at a time, from that writer's own image lists only,
    so a pair literally cannot reference two different writers or two
    different splits.
  - eligibility is enforced by skipping ineligible writers (per the
    Phase 3A manifest's pair_generation_eligible column) before any pair
    is built for them -- not by generating then filtering.
  - canonical ordering prevents reversed duplicates for free:
      * genuine_genuine pairs come from itertools.combinations on a
        SORTED genuine image_id list, which never produces both (a,b)
        and (b,a).
      * genuine_forgery pairs are label-asymmetric (image_1 is always
        the genuine "reference", image_2 is always the forgery "query"),
        so there is no reversed-pair case to begin with.
  - exact-duplicate self-pairs are excluded from genuine_genuine only.
    Phase 2's duplicate audit found ZERO exact-duplicate groups mix
    genuine and forgery labels (genuine_forgery_mixed_groups: 0), so a
    genuine_forgery pair can never be a same-image exact-duplicate pair
    by construction -- only genuine_genuine needs the filter.
"""

from __future__ import annotations

import itertools

import pandas as pd

GENUINE = "genuine"
FORGERY = "forgery"


def build_exact_duplicate_pairs(dup_audit_df: pd.DataFrame | None) -> set[frozenset]:
    """Returns {frozenset({image_a, image_b}), ...} for every pair of
    image_ids that are members of the SAME sha256_exact duplicate group.
    Used only to exclude degenerate (zero-distance) genuine_genuine pairs.
    """
    if dup_audit_df is None or len(dup_audit_df) == 0:
        return set()
    pairs: set[frozenset] = set()
    exact = dup_audit_df[dup_audit_df["hash_type"] == "sha256_exact"]
    for _, group in exact.groupby("hash_value"):
        ids = sorted(group["image_id"].dropna().unique().tolist())
        for a, b in itertools.combinations(ids, 2):
            pairs.add(frozenset((a, b)))
    return pairs


def make_pair_id(global_writer_id: str, pair_type: str, image_1_id: str, image_2_id: str) -> str:
    """Deterministic, human-readable, collision-free (image_ids are
    already globally unique) pair identifier.
    """
    return f"{global_writer_id}__{pair_type}__{image_1_id}__{image_2_id}"


def generate_writer_pairs(global_writer_id: str, dataset: str, split: str,
                          genuine_ids: list[str], forgery_ids: list[str],
                          exact_duplicate_pairs: set[frozenset]) -> list[dict]:
    """All candidate pairs for ONE writer. Sorting the input ID lists
    first is what makes both the enumeration order and the resulting
    pair_ids fully deterministic regardless of the input ordering.
    """
    genuine_sorted = sorted(genuine_ids)
    forgery_sorted = sorted(forgery_ids)
    rows: list[dict] = []

    skipped_exact_duplicate_pairs = 0
    for img1, img2 in itertools.combinations(genuine_sorted, 2):
        if frozenset((img1, img2)) in exact_duplicate_pairs:
            skipped_exact_duplicate_pairs += 1
            continue
        rows.append({
            "pair_id": make_pair_id(global_writer_id, "genuine_genuine", img1, img2),
            "image_1_id": img1, "image_2_id": img2,
            "global_writer_id": global_writer_id, "dataset": dataset, "split": split,
            "pair_label": 1, "pair_type": "genuine_genuine",
        })

    for img1 in genuine_sorted:
        for img2 in forgery_sorted:
            rows.append({
                "pair_id": make_pair_id(global_writer_id, "genuine_forgery", img1, img2),
                "image_1_id": img1, "image_2_id": img2,
                "global_writer_id": global_writer_id, "dataset": dataset, "split": split,
                "pair_label": 0, "pair_type": "genuine_forgery",
            })

    return rows


def generate_all_candidate_pairs(manifest_df: pd.DataFrame, image_metadata_df: pd.DataFrame,
                                 dup_audit_df: pd.DataFrame | None) -> pd.DataFrame:
    """Exhaustive candidate generation across every ELIGIBLE writer in
    the manifest. Writers with pair_generation_eligible == False (e.g.
    BHSIG_HINDI_123) are skipped entirely at the loop level -- they
    literally never enter pair construction, rather than being filtered
    out afterward. Deterministic writer processing order (sorted by
    global_writer_id) -- combined with the sorted image-ID ordering
    inside generate_writer_pairs -- makes the full output reproducible.
    """
    exact_duplicate_pairs = build_exact_duplicate_pairs(dup_audit_df)
    grouped_images = {gw: g for gw, g in image_metadata_df.groupby("global_writer_id")}

    eligible_writers = manifest_df[manifest_df["pair_generation_eligible"] == True]  # noqa: E712
    eligible_writers = eligible_writers.sort_values("global_writer_id")

    all_rows: list[dict] = []
    for row in eligible_writers.itertuples(index=False):
        gw = row.global_writer_id
        writer_images = grouped_images.get(gw)
        if writer_images is None:
            continue  # no images found for this writer -- nothing to pair
        genuine_ids = writer_images.loc[writer_images["signature_type"] == GENUINE, "image_id"].tolist()
        forgery_ids = writer_images.loc[writer_images["signature_type"] == FORGERY, "image_id"].tolist()
        all_rows.extend(
            generate_writer_pairs(gw, row.dataset, row.split, genuine_ids, forgery_ids, exact_duplicate_pairs)
        )

    columns = ["pair_id", "image_1_id", "image_2_id", "global_writer_id", "dataset", "split",
               "pair_label", "pair_type"]
    if not all_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(all_rows)[columns]
