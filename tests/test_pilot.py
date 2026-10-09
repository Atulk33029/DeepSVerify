import numpy as np
import pandas as pd
import pytest
from PIL import Image

from preprocessing.pilot import (
    select_pilot_sample,
    variant_name,
    run_pilot,
    summarize_pilot,
    select_comparison_examples,
)


PILOT_CFG = {
    "preprocessing_version": "test-pilot-v1",
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
    "pilot": {
        "intensity_variants": ["original", "flatten_bg"],
        "margin_fractions": [0.05, 0.10],
    },
}


@pytest.fixture
def synthetic_images_and_df(tmp_path):
    rows = []
    for dataset in ("CEDAR", "BHSIG_BENGALI"):
        for sig_type in ("genuine", "forgery"):
            for i in range(3):
                arr = np.full((60, 200), 255, dtype=np.uint8)
                arr[20:40, 50:150] = 0
                p = tmp_path / f"{dataset}_{sig_type}_{i}.png"
                Image.fromarray(arr, mode="L").save(p)
                rows.append({
                    "image_id": f"{dataset}_{sig_type}_{i}",
                    "dataset": dataset,
                    "global_writer_id": f"{dataset}_1",
                    "signature_type": sig_type,
                    "original_path": str(p),
                    "is_valid": True,
                })
    return pd.DataFrame(rows)


def test_variant_name_format():
    assert variant_name("original", 0.05) == "original_margin05"
    assert variant_name("flatten_bg", 0.10) == "flatten_bg_margin10"


def test_select_pilot_sample_deterministic(synthetic_images_and_df):
    s1 = select_pilot_sample(synthetic_images_and_df, n_per_group=2, seed=42)
    s2 = select_pilot_sample(synthetic_images_and_df, n_per_group=2, seed=42)
    assert list(s1["image_id"]) == list(s2["image_id"])


def test_select_pilot_sample_respects_group_cap(synthetic_images_and_df):
    sample = select_pilot_sample(synthetic_images_and_df, n_per_group=2, seed=42)
    counts = sample.groupby(["dataset", "signature_type"]).size()
    assert (counts <= 2).all()


def test_select_pilot_sample_excludes_invalid_rows(synthetic_images_and_df):
    df = synthetic_images_and_df.copy()
    df.loc[0, "is_valid"] = False
    excluded_id = df.loc[0, "image_id"]
    sample = select_pilot_sample(df, n_per_group=10, seed=42)
    assert excluded_id not in set(sample["image_id"])


def test_run_pilot_produces_all_variant_combinations(tmp_path, synthetic_images_and_df):
    sample = select_pilot_sample(synthetic_images_and_df, n_per_group=2, seed=42)
    output_root = tmp_path / "pilot_out"
    result = run_pilot(sample, PILOT_CFG, output_root)

    expected_variants = {"original_margin05", "original_margin10", "flatten_bg_margin05", "flatten_bg_margin10"}
    assert set(result["variant"].unique()) == expected_variants
    # every (image, variant) combination present exactly once
    assert len(result) == len(sample) * 4


def test_run_pilot_writes_output_files(tmp_path, synthetic_images_and_df):
    sample = select_pilot_sample(synthetic_images_and_df, n_per_group=1, seed=42)
    output_root = tmp_path / "pilot_out"
    result = run_pilot(sample, PILOT_CFG, output_root)
    for p in result["output_path"]:
        assert p is not None
        from pathlib import Path
        assert Path(p).exists()


def test_summarize_pilot_one_row_per_variant(tmp_path, synthetic_images_and_df):
    sample = select_pilot_sample(synthetic_images_and_df, n_per_group=2, seed=42)
    result = run_pilot(sample, PILOT_CFG, tmp_path / "pilot_out")
    summary = summarize_pilot(result)
    assert len(summary) == 4
    assert set(summary["variant"]) == {"original_margin05", "original_margin10",
                                        "flatten_bg_margin05", "flatten_bg_margin10"}
    assert (summary["n_images"] == len(sample)).all()


def test_select_comparison_examples_covers_each_dataset(synthetic_images_and_df):
    sample = select_pilot_sample(synthetic_images_and_df, n_per_group=3, seed=42)
    examples = select_comparison_examples(sample, n=4, seed=1)
    assert set(examples["dataset"]) == set(sample["dataset"].unique())


def test_select_comparison_examples_deterministic(synthetic_images_and_df):
    sample = select_pilot_sample(synthetic_images_and_df, n_per_group=3, seed=42)
    e1 = select_comparison_examples(sample, n=4, seed=1)
    e2 = select_comparison_examples(sample, n=4, seed=1)
    assert list(e1["image_id"]) == list(e2["image_id"])
