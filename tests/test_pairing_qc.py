import pandas as pd
import pytest

from pairing.qc import (
    check_no_split_crossing,
    check_writer_consistency,
    check_label_type_consistency,
    check_no_ineligible_writer_pairs,
    check_no_duplicate_pair_ids,
    check_no_reversed_duplicate_pairs,
    check_images_exist_and_match_metadata,
    check_bhsig_hindi_123_produces_zero_pairs,
    check_split_assignments_unchanged,
    run_full_qc,
)


@pytest.fixture
def clean_pairs_df():
    return pd.DataFrame([
        {"pair_id": "W1__gg__A__B", "image_1_id": "A", "image_2_id": "B",
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
        {"pair_id": "W1__gf__A__C", "image_1_id": "A", "image_2_id": "C",
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 0, "pair_type": "genuine_forgery"},
    ])


@pytest.fixture
def clean_image_metadata():
    return pd.DataFrame([
        {"image_id": "A", "global_writer_id": "W1", "signature_type": "genuine"},
        {"image_id": "B", "global_writer_id": "W1", "signature_type": "genuine"},
        {"image_id": "C", "global_writer_id": "W1", "signature_type": "forgery"},
    ])


@pytest.fixture
def clean_image_split_map():
    return pd.DataFrame([
        {"image_id": "A", "split": "train"},
        {"image_id": "B", "split": "train"},
        {"image_id": "C", "split": "train"},
    ])


@pytest.fixture
def clean_manifest():
    return pd.DataFrame([
        {"global_writer_id": "W1", "split": "train", "pair_generation_eligible": True},
    ])


# --------------------------------------------------------------------------
# 1-2: positive/negative pair validity, writer equality
# --------------------------------------------------------------------------

def test_check_writer_consistency_passes_clean(clean_pairs_df, clean_image_metadata):
    result = check_writer_consistency(clean_pairs_df, clean_image_metadata)
    assert result["passed"] is True


def test_check_writer_consistency_detects_mismatch(clean_image_metadata):
    bad = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "A", "image_2_id": "C",
         "global_writer_id": "WRONG_WRITER", "dataset": "CEDAR", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
    ])
    result = check_writer_consistency(bad, clean_image_metadata)
    assert result["passed"] is False


# --------------------------------------------------------------------------
# 4: split equality
# --------------------------------------------------------------------------

def test_check_no_split_crossing_passes_clean(clean_pairs_df, clean_image_split_map):
    result = check_no_split_crossing(clean_pairs_df, clean_image_split_map)
    assert result["passed"] is True


def test_check_no_split_crossing_detects_violation(clean_image_metadata):
    split_map = pd.DataFrame([{"image_id": "A", "split": "train"}, {"image_id": "C", "split": "test"}])
    bad = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "A", "image_2_id": "C",
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 0, "pair_type": "genuine_forgery"},
    ])
    result = check_no_split_crossing(bad, split_map)
    assert result["passed"] is False


# --------------------------------------------------------------------------
# label/type consistency
# --------------------------------------------------------------------------

def test_check_label_type_consistency_passes_clean(clean_pairs_df, clean_image_metadata):
    result = check_label_type_consistency(clean_pairs_df, clean_image_metadata)
    assert result["passed"] is True


def test_check_label_type_consistency_detects_wrong_label(clean_image_metadata):
    bad = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "A", "image_2_id": "B",
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 0, "pair_type": "genuine_genuine"},  # should be 1
    ])
    result = check_label_type_consistency(bad, clean_image_metadata)
    assert result["passed"] is False


def test_check_label_type_consistency_detects_wrong_image_types(clean_image_metadata):
    bad = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "C", "image_2_id": "A",  # forgery listed as image_1
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 0, "pair_type": "genuine_forgery"},
    ])
    result = check_label_type_consistency(bad, clean_image_metadata)
    assert result["passed"] is False


# --------------------------------------------------------------------------
# 5: eligibility filtering
# --------------------------------------------------------------------------

def test_check_no_ineligible_writer_pairs_passes_clean(clean_pairs_df, clean_manifest):
    result = check_no_ineligible_writer_pairs(clean_pairs_df, clean_manifest)
    assert result["passed"] is True


def test_check_no_ineligible_writer_pairs_detects_violation(clean_manifest):
    bad = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "X", "image_2_id": "Y",
         "global_writer_id": "BHSIG_HINDI_123", "dataset": "BHSIG_HINDI", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
    ])
    result = check_no_ineligible_writer_pairs(bad, clean_manifest)
    assert result["passed"] is False
    assert "BHSIG_HINDI_123" in result["violating_writers"]


# --------------------------------------------------------------------------
# 6-7: deterministic pair IDs, reversed-pair prevention
# --------------------------------------------------------------------------

def test_check_no_duplicate_pair_ids_passes_clean(clean_pairs_df):
    result = check_no_duplicate_pair_ids(clean_pairs_df)
    assert result["passed"] is True


def test_check_no_duplicate_pair_ids_detects_duplicate(clean_pairs_df):
    bad = pd.concat([clean_pairs_df, clean_pairs_df.iloc[[0]]], ignore_index=True)
    result = check_no_duplicate_pair_ids(bad)
    assert result["passed"] is False
    assert result["num_duplicates"] == 1


def test_check_no_reversed_duplicate_pairs_passes_clean(clean_pairs_df):
    result = check_no_reversed_duplicate_pairs(clean_pairs_df)
    assert result["passed"] is True


def test_check_no_reversed_duplicate_pairs_detects_reversal():
    bad = pd.DataFrame([
        {"pair_id": "p1", "image_1_id": "A", "image_2_id": "B",
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
        {"pair_id": "p2", "image_1_id": "B", "image_2_id": "A",  # same pair, reversed
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
    ])
    result = check_no_reversed_duplicate_pairs(bad)
    assert result["passed"] is False


# --------------------------------------------------------------------------
# 8: missing-file / referenced-image detection
# --------------------------------------------------------------------------

def test_check_images_exist_passes_clean(clean_pairs_df, clean_image_metadata):
    result = check_images_exist_and_match_metadata(clean_pairs_df, clean_image_metadata)
    assert result["passed"] is True


def test_check_images_exist_detects_missing_reference(clean_image_metadata):
    bad = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "A", "image_2_id": "GHOST_IMAGE",
         "global_writer_id": "W1", "dataset": "CEDAR", "split": "train", "pair_label": 0, "pair_type": "genuine_forgery"},
    ])
    result = check_images_exist_and_match_metadata(bad, clean_image_metadata)
    assert result["passed"] is False


# --------------------------------------------------------------------------
# 10: BHSIG_HINDI_123 exclusion
# --------------------------------------------------------------------------

def test_check_bhsig_hindi_123_passes_when_zero(clean_pairs_df):
    result = check_bhsig_hindi_123_produces_zero_pairs(clean_pairs_df)
    assert result["passed"] is True
    assert result["pairs_found"] == 0


def test_check_bhsig_hindi_123_fails_when_present(clean_pairs_df):
    bad = pd.concat([clean_pairs_df, pd.DataFrame([
        {"pair_id": "x", "image_1_id": "A", "image_2_id": "B",
         "global_writer_id": "BHSIG_HINDI_123", "dataset": "BHSIG_HINDI", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
    ])], ignore_index=True)
    result = check_bhsig_hindi_123_produces_zero_pairs(bad)
    assert result["passed"] is False
    assert result["pairs_found"] == 1


# --------------------------------------------------------------------------
# no Phase 3A split assignments changed
# --------------------------------------------------------------------------

def test_check_split_assignments_unchanged_passes_when_same():
    m1 = pd.DataFrame([{"global_writer_id": "W1", "split": "train"}])
    m2 = pd.DataFrame([{"global_writer_id": "W1", "split": "train"}])
    result = check_split_assignments_unchanged(m1, m2)
    assert result["passed"] is True


def test_check_split_assignments_unchanged_detects_change():
    m1 = pd.DataFrame([{"global_writer_id": "W1", "split": "test"}])  # changed!
    m2 = pd.DataFrame([{"global_writer_id": "W1", "split": "train"}])  # original
    result = check_split_assignments_unchanged(m1, m2)
    assert result["passed"] is False
    assert "W1" in result["changed_writers"]


# --------------------------------------------------------------------------
# run_full_qc: overall PASS/FAIL
# --------------------------------------------------------------------------

def test_run_full_qc_passes_on_clean_data(clean_pairs_df, clean_manifest, clean_image_metadata, clean_image_split_map):
    qc = run_full_qc(
        pairs_df=clean_pairs_df, manifest_df=clean_manifest, image_metadata_df=clean_image_metadata,
        image_split_map=clean_image_split_map, dup_audit_df=None, original_manifest_df=clean_manifest,
    )
    assert qc["status"] == "PASS"
    assert qc["critical_failures"] == []


def test_derive_known_data_characteristics_mechanical_no_hardcoded_writer():
    from pairing.report import derive_known_data_characteristics

    manifest = pd.DataFrame([
        {"global_writer_id": "BHSIG_HINDI_124", "dataset": "BHSIG_HINDI", "split": "train",
         "n_genuine": 48, "n_forgery": 30, "pair_generation_eligible": True},
        {"global_writer_id": "BHSIG_HINDI_1", "dataset": "BHSIG_HINDI", "split": "train",
         "n_genuine": 24, "n_forgery": 30, "pair_generation_eligible": True},
    ])
    qc_result = {
        "distribution_report": {
            "writers_with_unusually_high_count": {"BHSIG_HINDI_124": 2256},
            "writers_with_unusually_low_count": {},
        }
    }

    # No note provided -> still surfaces the writer, marked unverified.
    entries = derive_known_data_characteristics(qc_result, manifest)
    assert len(entries) == 1
    assert entries[0]["global_writer_id"] == "BHSIG_HINDI_124"
    assert entries[0]["n_genuine"] == 48
    assert entries[0]["status"] == "unverified_outlier"

    # With a verified note -> status flips, writer still derived from QC output, not hardcoded.
    entries2 = derive_known_data_characteristics(
        qc_result, manifest, verified_notes={"BHSIG_HINDI_124": "confirmed via raw-folder inspection"}
    )
    assert entries2[0]["status"] == "verified_source_data_characteristic"
    assert entries2[0]["note"] == "confirmed via raw-folder inspection"


def test_derive_known_data_characteristics_empty_when_no_outliers():
    from pairing.report import derive_known_data_characteristics
    manifest = pd.DataFrame([
        {"global_writer_id": "W1", "dataset": "CEDAR", "split": "train",
         "n_genuine": 24, "n_forgery": 24, "pair_generation_eligible": True},
    ])
    qc_result = {"distribution_report": {"writers_with_unusually_high_count": {}, "writers_with_unusually_low_count": {}}}
    assert derive_known_data_characteristics(qc_result, manifest) == []


def test_run_full_qc_fails_on_any_critical_violation(clean_manifest, clean_image_metadata, clean_image_split_map):
    bad_pairs = pd.DataFrame([
        {"pair_id": "x", "image_1_id": "A", "image_2_id": "B",
         "global_writer_id": "BHSIG_HINDI_123", "dataset": "BHSIG_HINDI", "split": "train", "pair_label": 1, "pair_type": "genuine_genuine"},
    ])
    qc = run_full_qc(
        pairs_df=bad_pairs, manifest_df=clean_manifest, image_metadata_df=clean_image_metadata,
        image_split_map=clean_image_split_map, dup_audit_df=None, original_manifest_df=clean_manifest,
    )
    assert qc["status"] == "FAIL"
    assert "no_ineligible_writer_produces_pairs" in qc["critical_failures"]
