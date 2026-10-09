import pandas as pd

from pairing.measurement import compute_theoretical_pair_counts, aggregate_pair_counts


def _manifest(rows):
    return pd.DataFrame(rows)


def test_compute_theoretical_pair_counts_formula():
    df = _manifest([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train",
         "n_genuine": 24, "n_forgery": 24, "pair_generation_eligible": True},
    ])
    result = compute_theoretical_pair_counts(df)
    row = result.iloc[0]
    assert row["possible_positive_pairs"] == 276  # C(24,2)
    assert row["possible_negative_pairs"] == 576  # 24*24


def test_compute_theoretical_pair_counts_ineligible_is_zero():
    df = _manifest([
        {"global_writer_id": "BHSIG_HINDI_123", "dataset": "BHSIG_HINDI", "split": "train",
         "n_genuine": 0, "n_forgery": 30, "pair_generation_eligible": False},
    ])
    result = compute_theoretical_pair_counts(df)
    row = result.iloc[0]
    assert row["possible_positive_pairs"] == 0
    assert row["possible_negative_pairs"] == 0


def test_compute_theoretical_pair_counts_small_n_genuine():
    df = _manifest([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train",
         "n_genuine": 1, "n_forgery": 5, "pair_generation_eligible": True},
    ])
    result = compute_theoretical_pair_counts(df)
    row = result.iloc[0]
    assert row["possible_positive_pairs"] == 0  # C(1,2) = 0
    assert row["possible_negative_pairs"] == 5


def test_aggregate_pair_counts_totals():
    df = _manifest([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train", "n_genuine": 24, "n_forgery": 24, "pair_generation_eligible": True},
        {"global_writer_id": "B", "dataset": "CEDAR", "split": "val", "n_genuine": 24, "n_forgery": 24, "pair_generation_eligible": True},
        {"global_writer_id": "C", "dataset": "BHSIG_HINDI", "split": "train", "n_genuine": 0, "n_forgery": 30, "pair_generation_eligible": False},
    ])
    counts = compute_theoretical_pair_counts(df)
    agg = aggregate_pair_counts(counts)

    assert agg["num_total_writers"] == 3
    assert agg["num_eligible_writers"] == 2
    assert agg["num_ineligible_writers"] == 1
    assert agg["ineligible_writer_ids"] == ["C"]
    assert agg["total_possible_positive_pairs"] == 276 * 2
    assert agg["total_possible_negative_pairs"] == 576 * 2


def test_aggregate_pair_counts_ratio():
    df = _manifest([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train", "n_genuine": 24, "n_forgery": 24, "pair_generation_eligible": True},
    ])
    counts = compute_theoretical_pair_counts(df)
    agg = aggregate_pair_counts(counts)
    assert agg["negative_to_positive_ratio_overall"] == round(576 / 276, 3)


def test_aggregate_pair_counts_by_split_and_dataset():
    df = _manifest([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train", "n_genuine": 24, "n_forgery": 24, "pair_generation_eligible": True},
        {"global_writer_id": "B", "dataset": "BHSIG_BENGALI", "split": "train", "n_genuine": 24, "n_forgery": 30, "pair_generation_eligible": True},
    ])
    counts = compute_theoretical_pair_counts(df)
    agg = aggregate_pair_counts(counts)
    assert "train" in agg["by_split"]
    assert "CEDAR" in agg["by_dataset"]
    assert "BHSIG_BENGALI" in agg["by_dataset"]


def test_aggregate_pair_counts_empty_eligible_handled():
    df = _manifest([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train", "n_genuine": 0, "n_forgery": 30, "pair_generation_eligible": False},
    ])
    counts = compute_theoretical_pair_counts(df)
    agg = aggregate_pair_counts(counts)
    assert agg["num_eligible_writers"] == 0
    assert agg["total_possible_positive_pairs"] == 0
    assert agg["negative_to_positive_ratio_overall"] is None
