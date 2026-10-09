from pathlib import Path

import pandas as pd
import pytest

from preprocessing.qc import run_full_batch_qc


QC_CFG = {"target": {"width": 40, "height": 20, "pad_value": 255}}


def _base_proc_row(image_id, **overrides):
    row = {
        "image_id": image_id, "dataset": "CEDAR", "writer_id": "1", "global_writer_id": "CEDAR_1",
        "signature_type": "genuine", "original_path": "x", "processed_path": f"{image_id}.png",
        "preprocessing_status": "ok", "processed_width": 40, "processed_height": 20,
        "out_min": 0, "out_max": 255, "aspect_distortion": 0.0, "flags": "",
        "foreground_fraction": 0.03, "content_fill_fraction": 0.5, "bbox_width": 50,
        "bbox_height": 25, "bbox_aspect_ratio": 2.0, "content_width": 384, "content_height": 192,
        "error": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def phase1_df():
    return pd.DataFrame([
        {"image_id": "A", "is_valid": True},
        {"image_id": "B", "is_valid": True},
        {"image_id": "C", "is_valid": True},
    ])


@pytest.fixture
def images_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("A.png", "B.png", "C.png"):
        (d / name).write_bytes(b"fake-png-bytes")
    return d


def _proc_df_with_paths(rows, images_dir):
    for r in rows:
        if r.get("processed_path"):
            r["processed_path"] = str(images_dir / r["processed_path"])
    return pd.DataFrame(rows)


def test_qc_passes_on_clean_batch(phase1_df, images_dir):
    rows = [_base_proc_row("A"), _base_proc_row("B"), _base_proc_row("C")]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "PASS"
    assert qc["critical_failures"] == []


def test_qc_fails_on_missing_output(phase1_df, images_dir):
    rows = [_base_proc_row("A"), _base_proc_row("B")]  # "C" missing entirely
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "FAIL"
    assert "no_missing_outputs" in qc["critical_failures"]
    assert "C" in qc["missing_outputs_sample"]


def test_qc_fails_on_duplicate_image_id(phase1_df, images_dir):
    rows = [_base_proc_row("A"), _base_proc_row("A"), _base_proc_row("B"), _base_proc_row("C")]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "FAIL"
    assert "no_duplicate_image_ids" in qc["critical_failures"]


def test_qc_fails_on_dimension_mismatch(phase1_df, images_dir):
    rows = [_base_proc_row("A", processed_width=99), _base_proc_row("B"), _base_proc_row("C")]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "FAIL"
    assert "all_ok_flagged_outputs_match_target_dimensions" in qc["critical_failures"]
    assert "A" in qc["dimension_mismatch_sample"]


def test_qc_fails_on_aspect_distortion_bug(phase1_df, images_dir):
    rows = [_base_proc_row("A", aspect_distortion=0.5), _base_proc_row("B"), _base_proc_row("C")]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "FAIL"
    assert "no_aspect_distortion_bug" in qc["critical_failures"]


def test_qc_tolerates_small_rounding_distortion_on_extreme_aspect_content(phase1_df, images_dir):
    # Realistic integer-pixel rounding noise on a very thin, small resized
    # crop (content_height=16 -> bound = max(0.03, (0.5/16)*2) ~= 0.0625)
    # must NOT trip the critical check -- only genuine stretching bugs
    # (distortion well beyond what rounding alone could produce) should.
    rows = [
        _base_proc_row("A", aspect_distortion=0.028, content_width=384, content_height=16),
        _base_proc_row("B"), _base_proc_row("C"),
    ]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "PASS"
    assert "no_aspect_distortion_bug" not in qc["critical_failures"]


def test_qc_still_catches_large_distortion_on_small_content(phase1_df, images_dir):
    # Even with the size-aware bound, a sufficiently large distortion
    # (here ~30%, far beyond any plausible rounding noise even at
    # content_height=16) must still be caught as a real bug.
    rows = [
        _base_proc_row("A", aspect_distortion=0.30, content_width=384, content_height=16),
        _base_proc_row("B"), _base_proc_row("C"),
    ]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "FAIL"
    assert "no_aspect_distortion_bug" in qc["critical_failures"]


def test_qc_informational_flags_do_not_fail_status(phase1_df, images_dir):
    rows = [
        _base_proc_row("A", flags="tiny_foreground", preprocessing_status="flagged"),
        _base_proc_row("B"), _base_proc_row("C"),
    ]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "PASS"  # informational flag, not critical
    assert qc["informational_flags"]["tiny_foreground"]["count"] == 1


def test_qc_failed_images_never_dropped_from_report(phase1_df, images_dir):
    rows = [
        _base_proc_row("A", preprocessing_status="failed", processed_path=None, error="file_not_found"),
        _base_proc_row("B"), _base_proc_row("C"),
    ]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["counts"]["status_failed"] == 1
    assert any(item["image_id"] == "A" for item in qc["failed_images_sample"])
    # a failed image's row still counts toward output_count_matches_valid_input_count
    assert qc["critical_checks"]["output_count_matches_valid_input_count"] is True


def test_qc_detects_extra_unreferenced_file_on_disk(phase1_df, images_dir):
    (images_dir / "EXTRA.png").write_bytes(b"stray")
    rows = [_base_proc_row("A"), _base_proc_row("B"), _base_proc_row("C")]
    proc_df = _proc_df_with_paths(rows, images_dir)
    qc = run_full_batch_qc(phase1_df, proc_df, images_dir, QC_CFG)
    assert qc["status"] == "FAIL"
    assert "no_unreferenced_extra_files_on_disk" in qc["critical_failures"]
    assert "EXTRA" in qc["extra_files_on_disk_sample"]
