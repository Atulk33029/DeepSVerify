"""
splitting/writer_split.py

Phase 3A core logic: decides which writers go into train/val/test.

Design:
  1. Build a writer-level summary (one row per global_writer_id) from the
     real Phase 1 metadata: dataset, n_images, n_genuine, n_forgery.
  2. Apportion target writer COUNTS per split, independently within each
     dataset, using the largest-remainder method (deterministic, exact --
     always sums to that dataset's writer count, no rounding drift).
     Doing this per-dataset (rather than globally) is what preserves each
     dataset's representation across splits without touching writer
     disjointness: every split gets roughly the configured fraction of
     each dataset's writers, not an arbitrary mix.
  3. Assign WHICH writers land in which split via a seeded deterministic
     shuffle of each dataset's writer list, then slice by the apportioned
     counts. A writer-disjoint assignment is structural here: every
     writer appears in the shuffled list exactly once, so it can only
     land in exactly one split slice.

Nothing here generates pairs or touches individual images beyond
counting; Phase 3B (pair generation) is a separate, later step.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def build_writer_summary(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """One row per global_writer_id: dataset, n_images, n_genuine, n_forgery.
    Uses only valid rows (is_valid == True) if that column is present.
    """
    df = metadata_df
    if "is_valid" in df.columns:
        df = df[df["is_valid"] == True]  # noqa: E712

    def _agg(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "dataset": group["dataset"].iloc[0],
            "n_images": len(group),
            "n_genuine": int((group["signature_type"] == "genuine").sum()),
            "n_forgery": int((group["signature_type"] == "forgery").sum()),
        })

    summary = df.groupby("global_writer_id").apply(_agg, include_groups=False)
    summary = summary.reset_index().sort_values("global_writer_id").reset_index(drop=True)
    for col in ("n_images", "n_genuine", "n_forgery"):
        summary[col] = summary[col].astype(int)
    return summary


def apportion_counts(n: int, fracs: list[float]) -> list[int]:
    """Largest-remainder apportionment: splits n items into len(fracs)
    integer buckets proportional to fracs, guaranteed to sum to exactly n.
    Deterministic: ties in the remainder go to the earliest bucket in
    `fracs`'s order, so the same (n, fracs) always produces the same
    result -- no randomness here, randomness only enters when deciding
    WHICH writers fill each bucket (see assign_splits).
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1.0, got {sum(fracs)}")
    raw = [n * f for f in fracs]
    floors = [int(x) for x in raw]
    remainder = n - sum(floors)
    remainders = sorted(range(len(fracs)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in remainders[:remainder]:
        floors[i] += 1
    return floors


def assign_splits(writer_summary: pd.DataFrame, train_frac: float, val_frac: float,
                  test_frac: float, seed: int) -> pd.DataFrame:
    """Returns writer_summary with an added 'split' column (train/val/test).
    Stratified by dataset (see module docstring); deterministic given the
    same input, fractions, and seed.
    """
    fracs = [train_frac, val_frac, test_frac]
    names = ["train", "val", "test"]
    result_parts = []

    for dataset, group in writer_summary.groupby("dataset"):
        writers = sorted(group["global_writer_id"].tolist())  # deterministic base order
        counts = apportion_counts(len(writers), fracs)

        rng = random.Random(f"{seed}:{dataset}")
        shuffled = writers[:]
        rng.shuffle(shuffled)

        assignment = {}
        idx = 0
        for name, count in zip(names, counts):
            for writer_id in shuffled[idx:idx + count]:
                assignment[writer_id] = name
            idx += count

        g = group.copy()
        g["split"] = g["global_writer_id"].map(assignment)
        result_parts.append(g)

    result = pd.concat(result_parts, ignore_index=True).sort_values("global_writer_id").reset_index(drop=True)
    if result["split"].isna().any():
        missing = result.loc[result["split"].isna(), "global_writer_id"].tolist()
        raise RuntimeError(f"Internal error: {len(missing)} writers were not assigned a split: {missing}")
    return result


def flag_insufficient_class_support(writer_split_df: pd.DataFrame, min_genuine: int, min_forgery: int) -> pd.DataFrame:
    """Returns the subset of writers that do NOT meet the minimum
    genuine/forgery counts needed to support pair generation later
    (e.g. >=2 genuine signatures to form a genuine-genuine pair, >=1
    forgery to form a genuine-forgery pair). Flags only -- never excludes
    or modifies the split assignment.
    """
    insufficient = writer_split_df[
        (writer_split_df["n_genuine"] < min_genuine) | (writer_split_df["n_forgery"] < min_forgery)
    ].copy()
    return insufficient


def _ineligibility_reason(n_genuine: int, n_forgery: int, min_genuine: int, min_forgery: int) -> str:
    genuine_short = n_genuine < min_genuine
    forgery_short = n_forgery < min_forgery
    if genuine_short and forgery_short:
        return f"insufficient_genuine_and_forgery (n_genuine={n_genuine}, n_forgery={n_forgery})"
    if genuine_short:
        return f"insufficient_genuine (n_genuine={n_genuine}, need >={min_genuine})"
    if forgery_short:
        return f"insufficient_forgery (n_forgery={n_forgery}, need >={min_forgery})"
    return ""


def add_pair_generation_eligibility(writer_split_df: pd.DataFrame, min_genuine: int, min_forgery: int) -> pd.DataFrame:
    """Adds two columns, mechanically derived from n_genuine/n_forgery
    against the configured thresholds -- no writer is ever hard-coded as
    a special case:
      - pair_generation_eligible (bool): True iff n_genuine >= min_genuine
        AND n_forgery >= min_forgery.
      - ineligibility_reason (str): empty when eligible; otherwise a
        human-readable reason naming which class fell short and by how
        much, so Phase 3B (and this report) can explain *why* a writer
        is excluded from pairing without recomputing anything.

    Never removes or reassigns a writer -- every row from writer_split_df
    is preserved as-is, with these two columns appended.
    """
    df = writer_split_df.copy()
    eligible = (df["n_genuine"] >= min_genuine) & (df["n_forgery"] >= min_forgery)
    df["pair_generation_eligible"] = eligible
    df["ineligibility_reason"] = [
        "" if elig else _ineligibility_reason(g, f, min_genuine, min_forgery)
        for elig, g, f in zip(eligible, df["n_genuine"], df["n_forgery"])
    ]
    return df


def build_image_split_map(metadata_df: pd.DataFrame, writer_split_df: pd.DataFrame) -> pd.DataFrame:
    """Joins the writer-level split assignment onto every individual
    image (by global_writer_id), producing an image_id -> split lookup
    table. This is metadata only -- it does not generate any pairs.
    """
    writer_to_split = writer_split_df.set_index("global_writer_id")["split"].to_dict()
    df = metadata_df.copy()
    df["split"] = df["global_writer_id"].map(writer_to_split)
    return df[["image_id", "dataset", "global_writer_id", "signature_type", "split"]]


def write_split_manifest(writer_split_df: pd.DataFrame, path: Path) -> None:
    """Writes the Phase 3A manifest. Requires writer_split_df to already
    carry 'pair_generation_eligible' / 'ineligibility_reason' (i.e.
    add_pair_generation_eligibility must be called first) -- this is
    checked explicitly rather than silently omitting the columns, since
    Phase 3B is meant to consume this field directly instead of
    rediscovering class availability itself.
    """
    required_cols = {"pair_generation_eligible", "ineligibility_reason"}
    missing = required_cols - set(writer_split_df.columns)
    if missing:
        raise ValueError(
            f"write_split_manifest requires {required_cols} on writer_split_df "
            f"(missing: {missing}) -- call add_pair_generation_eligibility() first."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["global_writer_id", "dataset", "split", "n_images", "n_genuine", "n_forgery",
               "pair_generation_eligible", "ineligibility_reason"]
    writer_split_df[columns].sort_values(["split", "dataset", "global_writer_id"]).to_csv(path, index=False)
