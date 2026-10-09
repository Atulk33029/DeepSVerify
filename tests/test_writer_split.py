import pandas as pd
import pytest

from splitting.writer_split import (
    build_writer_summary,
    apportion_counts,
    assign_splits,
    flag_insufficient_class_support,
    add_pair_generation_eligibility,
    build_image_split_map,
    write_split_manifest,
)


def _make_metadata(n_writers_per_dataset=10, genuine=24, forgery=24):
    rows = []
    for dataset in ("CEDAR", "BHSIG_BENGALI", "BHSIG_HINDI"):
        for w in range(1, n_writers_per_dataset + 1):
            gw = f"{dataset}_{w}"
            for i in range(genuine):
                rows.append({"image_id": f"{gw}_G{i}", "dataset": dataset, "global_writer_id": gw,
                             "signature_type": "genuine", "is_valid": True})
            for i in range(forgery):
                rows.append({"image_id": f"{gw}_F{i}", "dataset": dataset, "global_writer_id": gw,
                             "signature_type": "forgery", "is_valid": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# build_writer_summary
# --------------------------------------------------------------------------

def test_build_writer_summary_counts_correct():
    df = _make_metadata(n_writers_per_dataset=2, genuine=5, forgery=3)
    summary = build_writer_summary(df)
    assert len(summary) == 6  # 2 writers x 3 datasets
    row = summary[summary["global_writer_id"] == "CEDAR_1"].iloc[0]
    assert row["n_genuine"] == 5
    assert row["n_forgery"] == 3
    assert row["n_images"] == 8


def test_build_writer_summary_excludes_invalid_rows():
    df = _make_metadata(n_writers_per_dataset=1, genuine=2, forgery=2)
    df.loc[0, "is_valid"] = False
    summary = build_writer_summary(df)
    row = summary[summary["global_writer_id"] == "CEDAR_1"].iloc[0]
    assert row["n_images"] == 3  # one excluded


# --------------------------------------------------------------------------
# apportion_counts: exact, deterministic, largest-remainder
# --------------------------------------------------------------------------

def test_apportion_counts_sums_to_n():
    result = apportion_counts(55, [0.70, 0.15, 0.15])
    assert sum(result) == 55


def test_apportion_counts_exact_divisible_case():
    result = apportion_counts(100, [0.70, 0.15, 0.15])
    assert result == [70, 15, 15]


def test_apportion_counts_bengali_hindi_match_expected():
    assert apportion_counts(100, [0.70, 0.15, 0.15]) == [70, 15, 15]
    assert apportion_counts(160, [0.70, 0.15, 0.15]) == [112, 24, 24]


def test_apportion_counts_cedar_rounds_correctly():
    # 55 * [0.70, 0.15, 0.15] = [38.5, 8.25, 8.25]; floor sum = 54,
    # remainder 1 goes to the largest fractional remainder (train, 0.5)
    result = apportion_counts(55, [0.70, 0.15, 0.15])
    assert result == [39, 8, 8]
    assert sum(result) == 55


def test_apportion_counts_rejects_bad_fracs():
    with pytest.raises(ValueError):
        apportion_counts(100, [0.5, 0.3])  # doesn't sum to 1.0


def test_apportion_counts_deterministic():
    r1 = apportion_counts(55, [0.70, 0.15, 0.15])
    r2 = apportion_counts(55, [0.70, 0.15, 0.15])
    assert r1 == r2


# --------------------------------------------------------------------------
# assign_splits: exclusivity, determinism, dataset stratification
# --------------------------------------------------------------------------

def test_assign_splits_writer_exclusivity():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    result = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    assert result["global_writer_id"].duplicated().sum() == 0
    assert set(result["split"].unique()) <= {"train", "val", "test"}


def test_assign_splits_no_writer_dropped():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    result = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    assert set(result["global_writer_id"]) == set(summary["global_writer_id"])
    assert result["split"].isna().sum() == 0


def test_assign_splits_deterministic_same_seed():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    r1 = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    r2 = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    merged = r1.merge(r2, on="global_writer_id", suffixes=("_1", "_2"))
    assert (merged["split_1"] == merged["split_2"]).all()


def test_assign_splits_different_seed_can_differ():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    r1 = assign_splits(summary, 0.70, 0.15, 0.15, seed=1)
    r2 = assign_splits(summary, 0.70, 0.15, 0.15, seed=2)
    merged = r1.merge(r2, on="global_writer_id", suffixes=("_1", "_2"))
    # not a strict requirement that they MUST differ, but with 60 writers
    # across 3 datasets it would be a near-impossible coincidence for two
    # different seeds to produce an identical assignment
    assert (merged["split_1"] != merged["split_2"]).any()


def test_assign_splits_preserves_dataset_representation():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    result = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    table = result.groupby(["split", "dataset"]).size().unstack(fill_value=0)
    # every dataset must appear in every split with a nonzero count
    assert (table > 0).all().all()


def test_assign_splits_rejects_unequal_total_across_runs_gracefully():
    # Sanity: calling twice with the SAME dataframe object must not mutate
    # it in a way that breaks a second call (no hidden state).
    df = _make_metadata(n_writers_per_dataset=10)
    summary = build_writer_summary(df)
    assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    result2 = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    assert len(result2) == len(summary)


# --------------------------------------------------------------------------
# flag_insufficient_class_support
# --------------------------------------------------------------------------

def test_flag_insufficient_class_support_detects_zero_forgery():
    df = _make_metadata(n_writers_per_dataset=5, genuine=24, forgery=24)
    # knock out all forgery samples for one writer
    df = df[~((df["global_writer_id"] == "CEDAR_1") & (df["signature_type"] == "forgery"))]
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    insufficient = flag_insufficient_class_support(split_df, min_genuine=2, min_forgery=1)
    assert "CEDAR_1" in insufficient["global_writer_id"].values


def test_flag_insufficient_class_support_none_when_all_sufficient():
    df = _make_metadata(n_writers_per_dataset=5, genuine=24, forgery=24)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    insufficient = flag_insufficient_class_support(split_df, min_genuine=2, min_forgery=1)
    assert len(insufficient) == 0


def test_flag_insufficient_class_support_never_modifies_split_assignment():
    df = _make_metadata(n_writers_per_dataset=5, genuine=1, forgery=0)  # everyone insufficient
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    before = split_df["split"].tolist()
    flag_insufficient_class_support(split_df, min_genuine=2, min_forgery=1)
    assert split_df["split"].tolist() == before  # unchanged -- flag only, never excludes


# --------------------------------------------------------------------------
# build_image_split_map
# --------------------------------------------------------------------------

def test_build_image_split_map_every_image_gets_a_split():
    df = _make_metadata(n_writers_per_dataset=5, genuine=3, forgery=3)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    image_map = build_image_split_map(df, split_df)
    assert len(image_map) == len(df)
    assert image_map["split"].isna().sum() == 0


def test_build_image_split_map_matches_writer_assignment():
    df = _make_metadata(n_writers_per_dataset=5, genuine=3, forgery=3)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    image_map = build_image_split_map(df, split_df)

    writer_to_split = split_df.set_index("global_writer_id")["split"].to_dict()
    for row in image_map.itertuples(index=False):
        assert row.split == writer_to_split[row.global_writer_id]


# --------------------------------------------------------------------------
# add_pair_generation_eligibility (Phase 3A review follow-up)
# --------------------------------------------------------------------------

def _writer_df_with_a_zero_genuine_writer():
    """Mirrors the real finding: one writer (modeled on BHSIG_HINDI_123)
    has n_genuine=0, n_forgery=30; everyone else is comfortably eligible.
    """
    df = _make_metadata(n_writers_per_dataset=5, genuine=24, forgery=30)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    target = split_df.iloc[0]["global_writer_id"]
    split_df.loc[split_df["global_writer_id"] == target, "n_genuine"] = 0
    split_df.loc[split_df["global_writer_id"] == target, "n_forgery"] = 30
    return split_df, target


def test_eligibility_deterministic():
    split_df, _ = _writer_df_with_a_zero_genuine_writer()
    r1 = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    r2 = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    merged = r1.merge(r2, on="global_writer_id", suffixes=("_1", "_2"))
    assert (merged["pair_generation_eligible_1"] == merged["pair_generation_eligible_2"]).all()
    assert (merged["ineligibility_reason_1"] == merged["ineligibility_reason_2"]).all()


def test_writers_below_minimum_correctly_marked_ineligible():
    split_df, target = _writer_df_with_a_zero_genuine_writer()
    result = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    row = result[result["global_writer_id"] == target].iloc[0]
    assert row["pair_generation_eligible"] == False  # noqa: E712
    assert row["ineligibility_reason"] != ""


def test_writers_meeting_minimum_correctly_marked_eligible():
    split_df, target = _writer_df_with_a_zero_genuine_writer()
    result = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    others = result[result["global_writer_id"] != target]
    assert (others["pair_generation_eligible"] == True).all()  # noqa: E712
    assert (others["ineligibility_reason"] == "").all()


def test_zero_genuine_writer_ineligible_with_correct_reason():
    # Mirrors BHSIG_HINDI_123 exactly: n_genuine=0, n_forgery=30 -- the
    # real finding this whole set of changes was built to handle.
    split_df, target = _writer_df_with_a_zero_genuine_writer()
    result = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    row = result[result["global_writer_id"] == target].iloc[0]
    assert row["n_genuine"] == 0
    assert row["n_forgery"] == 30
    assert row["pair_generation_eligible"] == False  # noqa: E712
    assert "genuine" in row["ineligibility_reason"].lower()


def test_eligibility_never_removes_or_hardcodes_a_writer():
    split_df, target = _writer_df_with_a_zero_genuine_writer()
    result = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    # every writer from the input is still present -- eligibility is a
    # flag, never an exclusion
    assert set(result["global_writer_id"]) == set(split_df["global_writer_id"])
    assert len(result) == len(split_df)
    # the function must derive ineligibility purely from the counts, not
    # special-case any specific writer_id: renaming the flagged writer
    # and re-running must still correctly flag it under its new name
    renamed = split_df.copy()
    renamed["global_writer_id"] = renamed["global_writer_id"].replace({target: "SOME_OTHER_WRITER_NAME"})
    result2 = add_pair_generation_eligibility(renamed, min_genuine=2, min_forgery=1)
    row2 = result2[result2["global_writer_id"] == "SOME_OTHER_WRITER_NAME"].iloc[0]
    assert row2["pair_generation_eligible"] == False  # noqa: E712


def test_all_writers_remain_represented_after_eligibility_added():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    result = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    assert len(result) == len(summary)
    assert set(result["global_writer_id"]) == set(summary["global_writer_id"])


def test_writer_disjointness_unchanged_by_eligibility():
    df = _make_metadata(n_writers_per_dataset=20)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    result = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    assert result["global_writer_id"].duplicated().sum() == 0
    # split assignment itself must be byte-for-byte unchanged by adding eligibility columns
    assert (result["split"].values == split_df["split"].values).all()


def test_write_split_manifest_requires_eligibility_columns(tmp_path):
    df = _make_metadata(n_writers_per_dataset=5)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    with pytest.raises(ValueError):
        write_split_manifest(split_df, tmp_path / "manifest.csv")  # missing eligibility columns


def test_write_split_manifest_includes_eligibility_columns(tmp_path):
    df = _make_metadata(n_writers_per_dataset=5)
    summary = build_writer_summary(df)
    split_df = assign_splits(summary, 0.70, 0.15, 0.15, seed=42)
    split_df = add_pair_generation_eligibility(split_df, min_genuine=2, min_forgery=1)
    out_path = tmp_path / "manifest.csv"
    write_split_manifest(split_df, out_path)

    written = pd.read_csv(out_path)
    assert "pair_generation_eligible" in written.columns
    assert "ineligibility_reason" in written.columns
    assert len(written) == len(split_df)  # no writer silently dropped by writing the manifest
