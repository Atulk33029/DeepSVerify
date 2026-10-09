import pandas as pd
import pytest

from preprocessing.duplicate_audit import (
    load_duplicate_report,
    classify_groups,
    summarize_audit,
)


@pytest.fixture
def synthetic_metadata():
    rows = [
        # same-writer exact duplicate (benign)
        {"image_id": "CEDAR_000001", "dataset": "CEDAR", "writer_id": "1",
         "global_writer_id": "CEDAR_1", "signature_type": "genuine"},
        {"image_id": "CEDAR_000002", "dataset": "CEDAR", "writer_id": "1",
         "global_writer_id": "CEDAR_1", "signature_type": "genuine"},
        # cross-writer exact duplicate (should be flagged as leakage risk)
        {"image_id": "BHSIGB_000001", "dataset": "BHSIG_BENGALI", "writer_id": "001",
         "global_writer_id": "BHSIG_BENGALI_001", "signature_type": "genuine"},
        {"image_id": "BHSIGB_000002", "dataset": "BHSIG_BENGALI", "writer_id": "002",
         "global_writer_id": "BHSIG_BENGALI_002", "signature_type": "genuine"},
        # cross-dataset perceptual duplicate (should NOT be flagged as leakage)
        {"image_id": "BHSIGB_000010", "dataset": "BHSIG_BENGALI", "writer_id": "003",
         "global_writer_id": "BHSIG_BENGALI_003", "signature_type": "forgery"},
        {"image_id": "BHSIGH_000010", "dataset": "BHSIG_HINDI", "writer_id": "004",
         "global_writer_id": "BHSIG_HINDI_004", "signature_type": "forgery"},
        # genuine/forgery mixed same-writer exact duplicate
        {"image_id": "CEDAR_000050", "dataset": "CEDAR", "writer_id": "5",
         "global_writer_id": "CEDAR_5", "signature_type": "genuine"},
        {"image_id": "CEDAR_000051", "dataset": "CEDAR", "writer_id": "5",
         "global_writer_id": "CEDAR_5", "signature_type": "forgery"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_duplicate_report():
    rows = [
        {"hash_type": "sha256_exact", "hash_value": "hashA", "num_duplicates": 2,
         "image_ids": "CEDAR_000001;CEDAR_000002"},
        {"hash_type": "sha256_exact", "hash_value": "hashB", "num_duplicates": 2,
         "image_ids": "BHSIGB_000001;BHSIGB_000002"},
        {"hash_type": "phash_perceptual", "hash_value": "hashC", "num_duplicates": 2,
         "image_ids": "BHSIGB_000010;BHSIGH_000010"},
        {"hash_type": "sha256_exact", "hash_value": "hashD", "num_duplicates": 2,
         "image_ids": "CEDAR_000050;CEDAR_000051"},
        {"hash_type": "sha256_exact", "hash_value": "hashE", "num_duplicates": 2,
         "image_ids": "CEDAR_000001;CEDAR_999999"},  # CEDAR_999999 missing from metadata
    ]
    df = pd.DataFrame(rows)
    # Mirror what load_duplicate_report() does when reading from CSV --
    # classify_groups() expects image_ids to already be a list, not a
    # raw semicolon-joined string.
    df["image_ids"] = df["image_ids"].apply(lambda s: s.split(";"))
    return df


def test_load_duplicate_report_splits_image_ids(tmp_path):
    raw_df = pd.DataFrame([
        {"hash_type": "sha256_exact", "hash_value": "hashA", "num_duplicates": 2,
         "image_ids": "CEDAR_000001;CEDAR_000002"},
    ])
    p = tmp_path / "duplicate_report.csv"
    raw_df.to_csv(p, index=False)
    df = load_duplicate_report(p)
    assert df.iloc[0]["image_ids"] == ["CEDAR_000001", "CEDAR_000002"]


def test_classify_groups_same_writer_not_flagged(synthetic_duplicate_report, synthetic_metadata):
    audit = classify_groups(synthetic_duplicate_report, synthetic_metadata)
    row_a = audit[audit["hash_value"] == "hashA"]
    assert (row_a["group_same_writer"] == True).all()
    assert (row_a["group_leakage_risk"] == False).all()
    assert (row_a["group_label_composition"] == "genuine_genuine").all()


def test_classify_groups_cross_writer_exact_is_flagged(synthetic_duplicate_report, synthetic_metadata):
    audit = classify_groups(synthetic_duplicate_report, synthetic_metadata)
    row_b = audit[audit["hash_value"] == "hashB"]
    assert (row_b["group_same_writer"] == False).all()
    assert (row_b["group_leakage_risk"] == True).all()


def test_classify_groups_cross_dataset_perceptual_not_leakage_flagged(synthetic_duplicate_report, synthetic_metadata):
    audit = classify_groups(synthetic_duplicate_report, synthetic_metadata)
    row_c = audit[audit["hash_value"] == "hashC"]
    assert (row_c["group_same_dataset"] == False).all()
    # perceptual groups are never auto-flagged as leakage_risk, regardless of writer scope
    assert (row_c["group_leakage_risk"] == False).all()


def test_classify_groups_mixed_label_composition(synthetic_duplicate_report, synthetic_metadata):
    audit = classify_groups(synthetic_duplicate_report, synthetic_metadata)
    row_d = audit[audit["hash_value"] == "hashD"]
    assert (row_d["group_label_composition"] == "genuine_forgery_mixed").all()
    assert (row_d["group_same_writer"] == True).all()


def test_classify_groups_handles_missing_image_id(synthetic_duplicate_report, synthetic_metadata):
    audit = classify_groups(synthetic_duplicate_report, synthetic_metadata)
    row_e = audit[audit["hash_value"] == "hashE"]
    assert row_e["group_missing_image_ids"].iloc[0] == 1
    missing_row = row_e[row_e["image_id"] == "CEDAR_999999"]
    assert missing_row["group_label_composition"].iloc[0] == "MISSING_FROM_METADATA"


def test_summarize_audit_counts(synthetic_duplicate_report, synthetic_metadata):
    audit = classify_groups(synthetic_duplicate_report, synthetic_metadata)
    summary = summarize_audit(audit)

    assert summary["A_images_in_exact_duplicate_groups"] == 7  # unique image_ids across hashA/B/D/E (incl. the missing one, counted by id)
    assert summary["B_images_in_perceptual_duplicate_groups"] == 2

    exact = summary["C_exact_duplicate_group_breakdown"]
    assert exact["num_groups"] == 4  # hashA, hashB, hashD, hashE
    assert exact["cross_writer_groups"] == 1  # hashB only (hashE has a missing member -> same_writer becomes ambiguous but only 1 present writer -> True)

    assert summary["F_leakage_risk"]["num_flagged_groups"] == 1
    assert "hashB" in summary["F_leakage_risk"]["flagged_group_hash_values"]
    assert summary["missing_from_metadata"] == 1
