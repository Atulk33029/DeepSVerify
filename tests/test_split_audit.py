import pandas as pd
import pytest

from splitting.split_audit import (
    check_writer_exclusivity,
    check_no_missing_writers,
    check_dataset_representation,
    check_class_availability,
    check_duplicate_group_cross_split,
    build_leakage_report,
)


@pytest.fixture
def clean_writer_split_df():
    rows = []
    for dataset in ("CEDAR", "BHSIG_BENGALI", "BHSIG_HINDI"):
        for split, n in (("train", 7), ("val", 2), ("test", 2)):
            for i in range(n):
                rows.append({
                    "global_writer_id": f"{dataset}_{split}_{i}", "dataset": dataset, "split": split,
                    "n_images": 48, "n_genuine": 24, "n_forgery": 24,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def clean_image_split_map(clean_writer_split_df):
    rows = []
    for w in clean_writer_split_df.itertuples(index=False):
        for i in range(w.n_genuine):
            rows.append({"image_id": f"{w.global_writer_id}_G{i}", "dataset": w.dataset,
                         "global_writer_id": w.global_writer_id, "signature_type": "genuine", "split": w.split})
        for i in range(w.n_forgery):
            rows.append({"image_id": f"{w.global_writer_id}_F{i}", "dataset": w.dataset,
                         "global_writer_id": w.global_writer_id, "signature_type": "forgery", "split": w.split})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# writer exclusivity / missing writers
# --------------------------------------------------------------------------

def test_check_writer_exclusivity_passes_on_clean_data(clean_writer_split_df):
    result = check_writer_exclusivity(clean_writer_split_df)
    assert result["passed"] is True
    assert result["duplicated_writers"] == []


def test_check_writer_exclusivity_detects_duplicate():
    df = pd.DataFrame([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train"},
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "test"},  # same writer, two rows
        {"global_writer_id": "B", "dataset": "CEDAR", "split": "val"},
    ])
    result = check_writer_exclusivity(df)
    assert result["passed"] is False
    assert "A" in result["duplicated_writers"]


def test_check_no_missing_writers_passes(clean_writer_split_df):
    original = set(clean_writer_split_df["global_writer_id"])
    result = check_no_missing_writers(original, clean_writer_split_df)
    assert result["passed"] is True


def test_check_no_missing_writers_detects_dropped_writer(clean_writer_split_df):
    original = set(clean_writer_split_df["global_writer_id"]) | {"GHOST_WRITER"}
    result = check_no_missing_writers(original, clean_writer_split_df)
    assert result["passed"] is False
    assert "GHOST_WRITER" in result["missing_writers"]


# --------------------------------------------------------------------------
# dataset representation
# --------------------------------------------------------------------------

def test_check_dataset_representation_passes_on_clean_data(clean_writer_split_df):
    result = check_dataset_representation(clean_writer_split_df)
    assert result["passed"] is True
    assert result["zero_representation_cells"] == []


def test_check_dataset_representation_detects_zero_cell():
    df = pd.DataFrame([
        {"global_writer_id": "A", "dataset": "CEDAR", "split": "train"},
        {"global_writer_id": "B", "dataset": "CEDAR", "split": "val"},
        {"global_writer_id": "C", "dataset": "BHSIG_BENGALI", "split": "train"},
        # BHSIG_BENGALI has no writer in val or test -> zero cells
    ])
    result = check_dataset_representation(df)
    assert result["passed"] is False
    assert len(result["zero_representation_cells"]) > 0


# --------------------------------------------------------------------------
# class availability
# --------------------------------------------------------------------------

def test_check_class_availability_passes_when_sufficient(clean_writer_split_df):
    result = check_class_availability(clean_writer_split_df, min_genuine=2, min_forgery=1)
    assert result["passed"] is True
    assert result["num_writers_insufficient"] == 0


def test_check_class_availability_detects_insufficient(clean_writer_split_df):
    df = clean_writer_split_df.copy()
    df.loc[0, "n_forgery"] = 0
    result = check_class_availability(df, min_genuine=2, min_forgery=1)
    assert result["passed"] is False
    assert result["num_writers_insufficient"] == 1


# --------------------------------------------------------------------------
# duplicate-group cross-split checks
# --------------------------------------------------------------------------

def test_check_duplicate_group_cross_split_same_split_not_flagged(clean_image_split_map):
    some_writer_images = clean_image_split_map[clean_image_split_map["global_writer_id"] == "CEDAR_train_0"]["image_id"].tolist()[:2]
    dup_df = pd.DataFrame([
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": some_writer_images[0]},
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": some_writer_images[1]},
    ])
    result = check_duplicate_group_cross_split(dup_df, clean_image_split_map, "sha256_exact")
    assert result["groups_spanning_multiple_splits"] == 0


def test_check_duplicate_group_cross_split_detects_spanning_group(clean_image_split_map):
    train_image = clean_image_split_map[clean_image_split_map["split"] == "train"]["image_id"].iloc[0]
    test_image = clean_image_split_map[clean_image_split_map["split"] == "test"]["image_id"].iloc[0]
    dup_df = pd.DataFrame([
        {"hash_type": "phash_perceptual", "hash_value": "h1", "image_id": train_image},
        {"hash_type": "phash_perceptual", "hash_value": "h1", "image_id": test_image},
    ])
    result = check_duplicate_group_cross_split(dup_df, clean_image_split_map, "phash_perceptual")
    assert result["groups_spanning_multiple_splits"] == 1
    assert result["spanning_examples"][0]["hash_value"] == "h1"


# --------------------------------------------------------------------------
# build_leakage_report: overall PASS/FAIL logic
# --------------------------------------------------------------------------

def test_build_leakage_report_passes_on_clean_data(clean_writer_split_df, clean_image_split_map):
    original = set(clean_writer_split_df["global_writer_id"])
    dup_df = pd.DataFrame(columns=["hash_type", "hash_value", "image_id"])
    report = build_leakage_report(
        writer_split_df=clean_writer_split_df, original_writer_ids=original,
        image_split_map=clean_image_split_map, dup_audit_df=dup_df,
        min_genuine=2, min_forgery=1,
    )
    assert report["status"] == "PASS"
    assert report["critical_failures"] == []


def test_build_leakage_report_fails_on_exact_duplicate_spanning_splits(clean_writer_split_df, clean_image_split_map):
    original = set(clean_writer_split_df["global_writer_id"])
    train_image = clean_image_split_map[clean_image_split_map["split"] == "train"]["image_id"].iloc[0]
    test_image = clean_image_split_map[clean_image_split_map["split"] == "test"]["image_id"].iloc[0]
    dup_df = pd.DataFrame([
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": train_image},
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": test_image},
    ])
    report = build_leakage_report(
        writer_split_df=clean_writer_split_df, original_writer_ids=original,
        image_split_map=clean_image_split_map, dup_audit_df=dup_df,
        min_genuine=2, min_forgery=1,
    )
    assert report["status"] == "FAIL"
    assert "no_exact_duplicate_group_spans_multiple_splits" in report["critical_failures"]


def test_known_limitations_names_each_insufficient_writer_explicitly(clean_writer_split_df, clean_image_split_map):
    # Mirrors the real BHSIG_HINDI_123 finding: one writer with n_genuine=0.
    df = clean_writer_split_df.copy()
    target_idx = df.index[0]
    target_id = df.loc[target_idx, "global_writer_id"]
    df.loc[target_idx, "n_genuine"] = 0

    original = set(df["global_writer_id"])
    dup_df = pd.DataFrame(columns=["hash_type", "hash_value", "image_id"])
    report = build_leakage_report(
        writer_split_df=df, original_writer_ids=original,
        image_split_map=clean_image_split_map, dup_audit_df=dup_df,
        min_genuine=2, min_forgery=1,
    )

    assert len(report["known_limitations"]) == 1
    finding = report["known_limitations"][0]
    assert finding["global_writer_id"] == target_id
    assert finding["n_genuine"] == 0
    assert finding["pair_generation_eligible"] is False
    assert "genuine" in finding["reason"].lower()
    # class_availability itself is informational, not a critical failure --
    # this is a known, reported, real data characteristic, not a bug
    assert report["status"] == "PASS"
    assert "class_availability" not in report["critical_checks"]


def test_known_limitations_empty_when_all_writers_sufficient(clean_writer_split_df, clean_image_split_map):
    original = set(clean_writer_split_df["global_writer_id"])
    dup_df = pd.DataFrame(columns=["hash_type", "hash_value", "image_id"])
    report = build_leakage_report(
        writer_split_df=clean_writer_split_df, original_writer_ids=original,
        image_split_map=clean_image_split_map, dup_audit_df=dup_df,
        min_genuine=2, min_forgery=1,
    )
    assert report["known_limitations"] == []


def test_exact_duplicate_guarantee_unaffected_by_ineligible_writer(clean_writer_split_df, clean_image_split_map):
    # An ineligible (n_genuine=0) writer must not change how exact-duplicate
    # cross-split checking behaves -- these are orthogonal concerns.
    df = clean_writer_split_df.copy()
    df.loc[df.index[0], "n_genuine"] = 0

    writer_images = clean_image_split_map[clean_image_split_map["global_writer_id"] == df.iloc[1]["global_writer_id"]]["image_id"].tolist()[:2]
    dup_df = pd.DataFrame([
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": writer_images[0]},
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": writer_images[1]},
    ])
    original = set(df["global_writer_id"])
    report = build_leakage_report(
        writer_split_df=df, original_writer_ids=original,
        image_split_map=clean_image_split_map, dup_audit_df=dup_df,
        min_genuine=2, min_forgery=1,
    )
    assert report["exact_duplicate_cross_split"]["groups_spanning_multiple_splits"] == 0
    assert report["critical_checks"]["no_exact_duplicate_group_spans_multiple_splits"] is True


def test_writer_disjointness_unaffected_by_ineligible_writer(clean_writer_split_df, clean_image_split_map):
    df = clean_writer_split_df.copy()
    df.loc[df.index[0], "n_genuine"] = 0
    result = check_writer_exclusivity(df)
    assert result["passed"] is True
    assert result["total_distinct_writers"] == len(clean_writer_split_df)


def test_build_leakage_report_perceptual_spanning_does_not_fail(clean_writer_split_df, clean_image_split_map):
    original = set(clean_writer_split_df["global_writer_id"])
    train_image = clean_image_split_map[clean_image_split_map["split"] == "train"]["image_id"].iloc[0]
    test_image = clean_image_split_map[clean_image_split_map["split"] == "test"]["image_id"].iloc[0]
    dup_df = pd.DataFrame([
        {"hash_type": "phash_perceptual", "hash_value": "h1", "image_id": train_image},
        {"hash_type": "phash_perceptual", "hash_value": "h1", "image_id": test_image},
    ])
    report = build_leakage_report(
        writer_split_df=clean_writer_split_df, original_writer_ids=original,
        image_split_map=clean_image_split_map, dup_audit_df=dup_df,
        min_genuine=2, min_forgery=1,
    )
    assert report["status"] == "PASS"  # perceptual spanning is informational only
    assert report["perceptual_duplicate_cross_split"]["groups_spanning_multiple_splits"] == 1
