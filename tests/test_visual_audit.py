import pandas as pd
import pytest

from preprocessing.visual_audit import (
    select_typical_examples,
    select_difficult_examples,
    build_audit_manifest,
)


@pytest.fixture
def synthetic_proc_df():
    rows = []
    datasets = ["CEDAR", "BHSIG_BENGALI", "BHSIG_HINDI"]
    for dataset in datasets:
        for sig_type in ("genuine", "forgery"):
            for i in range(4):
                rows.append({
                    "image_id": f"{dataset}_{sig_type}_{i}",
                    "dataset": dataset, "signature_type": sig_type,
                    "original_path": f"/raw/{dataset}_{sig_type}_{i}.png",
                    "processed_path": f"/proc/{dataset}_{sig_type}_{i}.png",
                    "preprocessing_status": "ok", "flags": "",
                    "foreground_fraction": 0.01 + 0.01 * i,
                    "content_fill_fraction": 0.3 + 0.05 * i,
                    "bbox_width": 50 + i * 10, "bbox_height": 20 + i * 2,
                    "bbox_aspect_ratio": (50 + i * 10) / (20 + i * 2),
                    "out_mean": 240 + i,
                })
    # one explicitly flagged image per flag type
    for flag in ["blank_output", "tiny_foreground", "extreme_aspect", "margin_clamped"]:
        rows.append({
            "image_id": f"FLAGGED_{flag}",
            "dataset": "CEDAR", "signature_type": "genuine",
            "original_path": "/raw/x.png", "processed_path": "/proc/x.png",
            "preprocessing_status": "flagged", "flags": flag,
            "foreground_fraction": 0.001, "content_fill_fraction": 0.01,
            "bbox_width": 5, "bbox_height": 5, "bbox_aspect_ratio": 1.0,
            "out_mean": 255,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_dup_audit_df():
    return pd.DataFrame([
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": "CEDAR_genuine_0"},
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": "CEDAR_genuine_1"},
        {"hash_type": "phash_perceptual", "hash_value": "h2", "image_id": "BHSIG_BENGALI_genuine_0"},
    ])


def test_select_typical_examples_covers_all_six_groups(synthetic_proc_df):
    selected = select_typical_examples(synthetic_proc_df, seed=42, n_per_group=2)
    expected_keys = {
        "typical__CEDAR__genuine", "typical__CEDAR__forgery",
        "typical__BHSIG_BENGALI__genuine", "typical__BHSIG_BENGALI__forgery",
        "typical__BHSIG_HINDI__genuine", "typical__BHSIG_HINDI__forgery",
    }
    assert expected_keys.issubset(set(selected.keys()))
    for key in expected_keys:
        assert len(selected[key]) == 2


def test_select_typical_examples_excludes_flagged(synthetic_proc_df):
    selected = select_typical_examples(synthetic_proc_df, seed=42, n_per_group=10)
    all_selected = {iid for ids in selected.values() for iid in ids}
    assert "FLAGGED_blank_output" not in all_selected


def test_select_typical_examples_deterministic(synthetic_proc_df):
    s1 = select_typical_examples(synthetic_proc_df, seed=42, n_per_group=2)
    s2 = select_typical_examples(synthetic_proc_df, seed=42, n_per_group=2)
    assert s1 == s2


def test_select_difficult_examples_finds_duplicate_group_member(synthetic_proc_df, synthetic_dup_audit_df):
    selected = select_difficult_examples(synthetic_proc_df, synthetic_dup_audit_df, seed=1)
    assert len(selected["duplicate_group_member"]) > 0
    assert set(selected["duplicate_group_member"]).issubset(
        set(synthetic_dup_audit_df[synthetic_dup_audit_df["hash_type"] == "sha256_exact"]["image_id"])
    )


def test_select_difficult_examples_handles_no_duplicates_gracefully(synthetic_proc_df):
    selected = select_difficult_examples(synthetic_proc_df, None, seed=1)
    assert selected["duplicate_group_member"] == []


def test_select_difficult_examples_finds_qc_flagged(synthetic_proc_df):
    selected = select_difficult_examples(synthetic_proc_df, None, seed=1)
    assert "FLAGGED_blank_output" in selected["qc_flagged_examples"]
    assert "FLAGGED_tiny_foreground" in selected["qc_flagged_examples"]


def test_select_difficult_examples_largest_smallest_differ(synthetic_proc_df):
    selected = select_difficult_examples(synthetic_proc_df, None, seed=1)
    assert set(selected["largest_signature"]).isdisjoint(set(selected["smallest_signature"]))


def test_build_audit_manifest_combines_all_categories(synthetic_proc_df, synthetic_dup_audit_df):
    manifest = build_audit_manifest(synthetic_proc_df, synthetic_dup_audit_df, seed=42)
    assert "typical__CEDAR__genuine" in manifest
    assert "largest_signature" in manifest
    assert "duplicate_group_member" in manifest
    assert "qc_flagged_examples" in manifest
    # internal bookkeeping key should not leak into the flat manifest
    assert "qc_flagged_examples_by_type" not in manifest
