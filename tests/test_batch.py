import numpy as np
import pandas as pd
import pytest
from PIL import Image

from preprocessing.batch import run_full_batch, FINAL_METADATA_COLUMNS


BATCH_CFG = {
    "preprocessing_version": "test-batch-v1",
    "target": {"width": 40, "height": 20, "pad_value": 255},
    "interpolation": {"downscale": "box", "upscale": "bicubic"},
    "intensity": {"flatten_min_bg_level": 100},
    "flags": {
        "polarity_suspect_fraction": 0.5,
        "tiny_foreground_fraction": 0.003,
        "extreme_aspect_low": 0.3,
        "extreme_aspect_high": 10.0,
        "heavy_upscale_factor": 3.0,
    },
    "locked": {"intensity_variant": "flatten_bg", "margin_fraction": 0.10},
}


@pytest.fixture
def synthetic_batch_df(tmp_path):
    rows = []
    for i in range(5):
        arr = np.full((60, 200), 255, dtype=np.uint8)
        arr[20:40, 50:150] = 0
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr, mode="L").save(p)
        rows.append({
            "image_id": f"IMG_{i:03d}", "dataset": "CEDAR", "writer_id": "1",
            "global_writer_id": "CEDAR_1", "signature_type": "genuine",
            "original_path": str(p), "is_valid": True,
        })
    # one invalid row -- should be skipped entirely
    rows.append({
        "image_id": "IMG_INVALID", "dataset": "CEDAR", "writer_id": "1",
        "global_writer_id": "CEDAR_1", "signature_type": "genuine",
        "original_path": "does/not/matter", "is_valid": False,
    })
    # one valid row whose file is actually missing -> should be "failed", not dropped
    rows.append({
        "image_id": "IMG_MISSING", "dataset": "CEDAR", "writer_id": "2",
        "global_writer_id": "CEDAR_2", "signature_type": "forgery",
        "original_path": str(tmp_path / "nonexistent.png"), "is_valid": True,
    })
    return pd.DataFrame(rows)


def test_run_full_batch_skips_invalid_rows(tmp_path, synthetic_batch_df):
    images_dir = tmp_path / "out_images"
    metadata_csv = tmp_path / "metadata_out.csv"
    out_df, stats = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=False)
    assert "IMG_INVALID" not in set(out_df["image_id"])
    assert stats["total_valid_input_images"] == 6  # 5 good + 1 missing-file, both is_valid=True


def test_run_full_batch_never_drops_failed_images(tmp_path, synthetic_batch_df):
    images_dir = tmp_path / "out_images"
    metadata_csv = tmp_path / "metadata_out.csv"
    out_df, stats = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=False)
    missing_row = out_df[out_df["image_id"] == "IMG_MISSING"]
    assert len(missing_row) == 1
    assert missing_row["preprocessing_status"].iloc[0] == "failed"
    assert stats["failed"] == 1


def test_run_full_batch_output_schema(tmp_path, synthetic_batch_df):
    images_dir = tmp_path / "out_images"
    metadata_csv = tmp_path / "metadata_out.csv"
    out_df, _ = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=False)
    assert list(out_df.columns) == FINAL_METADATA_COLUMNS


def test_run_full_batch_writes_png_files(tmp_path, synthetic_batch_df):
    images_dir = tmp_path / "out_images"
    metadata_csv = tmp_path / "metadata_out.csv"
    out_df, _ = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=False)
    ok_rows = out_df[out_df["preprocessing_status"].isin(["ok", "flagged"])]
    for p in ok_rows["processed_path"]:
        from pathlib import Path
        assert Path(p).exists()


def test_run_full_batch_resume_skips_unchanged_images(tmp_path, synthetic_batch_df):
    images_dir = tmp_path / "out_images"
    metadata_csv = tmp_path / "metadata_out.csv"

    out_df1, stats1 = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=False)
    assert stats1["newly_processed"] == 6  # 5 ok + 1 failed, all newly attempted

    out_df2, stats2 = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=True)
    # The 5 successfully processed images should be resumed (skipped);
    # the previously-failed one has no output file, so it's retried.
    assert stats2["resumed_from_previous_run"] == 5
    assert stats2["newly_processed"] == 1


def test_run_full_batch_resume_reprocesses_on_config_change(tmp_path, synthetic_batch_df):
    images_dir = tmp_path / "out_images"
    metadata_csv = tmp_path / "metadata_out.csv"
    run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir, metadata_csv, resume=False)

    changed_cfg = dict(BATCH_CFG)
    changed_cfg["locked"] = {"intensity_variant": "flatten_bg", "margin_fraction": 0.20}  # different margin
    out_df2, stats2 = run_full_batch(synthetic_batch_df, changed_cfg, images_dir, metadata_csv, resume=True)
    # config_fingerprint changed -> nothing should be resumed
    assert stats2["resumed_from_previous_run"] == 0


def test_run_full_batch_deterministic_rerun_from_scratch(tmp_path, synthetic_batch_df):
    images_dir1 = tmp_path / "out1"
    images_dir2 = tmp_path / "out2"
    out1, _ = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir1, tmp_path / "m1.csv", resume=False)
    out2, _ = run_full_batch(synthetic_batch_df, BATCH_CFG, images_dir2, tmp_path / "m2.csv", resume=False)
    ok1 = out1[out1["preprocessing_status"] == "ok"].sort_values("image_id")["output_sha256"].tolist()
    ok2 = out2[out2["preprocessing_status"] == "ok"].sort_values("image_id")["output_sha256"].tolist()
    assert ok1 == ok2
