import pandas as pd

from preprocessing.eda import compute_metadata_stats, _describe


def _synthetic_metadata_df():
    rows = [
        {"image_id": "CEDAR_000001", "dataset": "CEDAR", "signature_type": "genuine",
         "width": 300, "height": 150, "mode": "L", "extension": ".png",
         "file_size_bytes": 5000, "is_valid": True},
        {"image_id": "CEDAR_000002", "dataset": "CEDAR", "signature_type": "forgery",
         "width": 320, "height": 160, "mode": "L", "extension": ".png",
         "file_size_bytes": 5200, "is_valid": True},
        {"image_id": "BHSIGB_000001", "dataset": "BHSIG_BENGALI", "signature_type": "genuine",
         "width": 200, "height": 100, "mode": "L", "extension": ".tif",
         "file_size_bytes": 8000, "is_valid": True},
        {"image_id": "BHSIGB_000002", "dataset": "BHSIG_BENGALI", "signature_type": "forgery",
         "width": None, "height": None, "mode": None, "extension": ".tif",
         "file_size_bytes": None, "is_valid": False},
    ]
    return pd.DataFrame(rows)


def test_describe_handles_empty_series():
    result = _describe(pd.Series([], dtype=float))
    assert result == {"count": 0}


def test_describe_basic_stats():
    result = _describe(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert result["count"] == 5
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["mean"] == 3.0


def test_compute_metadata_stats_excludes_invalid_rows():
    df = _synthetic_metadata_df()
    stats = compute_metadata_stats(df)
    assert stats["overall"]["total_rows"] == 4
    assert stats["overall"]["valid_rows"] == 3  # the invalid BHSIGB_000002 row excluded


def test_compute_metadata_stats_per_dataset_breakdown():
    df = _synthetic_metadata_df()
    stats = compute_metadata_stats(df)
    assert "CEDAR" in stats["by_dataset"]
    assert stats["by_dataset"]["CEDAR"]["count"] == 2
    assert stats["by_dataset"]["BHSIG_BENGALI"]["count"] == 1


def test_compute_metadata_stats_aspect_ratio_computed():
    df = _synthetic_metadata_df()
    stats = compute_metadata_stats(df)
    cedar = stats["by_dataset"]["CEDAR"]
    # width 300/height 150 = 2.0 ; width 320/height 160 = 2.0
    assert abs(cedar["aspect_ratio"]["mean"] - 2.0) < 1e-6
