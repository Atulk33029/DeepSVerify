import numpy as np
import pytest
from PIL import Image

from preprocessing.transforms import (
    load_grayscale,
    detect_foreground,
    crop_with_margin,
    resize_and_pad,
    preprocess_image,
    preprocess_gray,
)


TEST_CFG = {
    "preprocessing_version": "test-v1",
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
}


def _make_bilevel_image(path, size=(200, 60), ink_box=(50, 20, 150, 40)):
    """White background, black rectangle 'ink' at ink_box (x0,y0,x1,y1)."""
    arr = np.full((size[1], size[0]), 255, dtype=np.uint8)
    x0, y0, x1, y1 = ink_box
    arr[y0:y1, x0:x1] = 0
    Image.fromarray(arr, mode="L").save(path)
    return arr


def _make_grayscale_image(path, size=(200, 60), bg=235, ink_box=(50, 20, 150, 40), ink_val=100):
    arr = np.full((size[1], size[0]), bg, dtype=np.uint8)
    x0, y0, x1, y1 = ink_box
    arr[y0:y1, x0:x1] = ink_val
    Image.fromarray(arr, mode="L").save(path)
    return arr


# --------------------------------------------------------------------------
# load_grayscale: mode conversion
# --------------------------------------------------------------------------

def test_load_grayscale_plain_L(tmp_path):
    p = tmp_path / "img.png"
    _make_grayscale_image(p)
    gray, info = load_grayscale(p)
    assert gray.dtype == np.uint8
    assert info["original_mode"] == "L"
    assert info["alpha_composited"] is False


def test_load_grayscale_rgba_composited_on_white(tmp_path):
    p = tmp_path / "img.png"
    arr = np.zeros((30, 30, 4), dtype=np.uint8)
    arr[..., 3] = 0  # fully transparent
    arr[10:20, 10:20, 3] = 255  # opaque black square in the middle
    Image.fromarray(arr, mode="RGBA").save(p)

    gray, info = load_grayscale(p)
    assert info["original_mode"] == "RGBA"
    assert info["alpha_composited"] is True
    assert info["alpha_min"] == 0
    # transparent region should have become white (255), not black
    assert gray[0, 0] == 255
    # opaque region should stay black
    assert gray[15, 15] == 0


def test_load_grayscale_palette_mode(tmp_path):
    p = tmp_path / "img.png"
    img = Image.new("P", (30, 30))
    img.putpixel((5, 5), 1)
    img.save(p)
    gray, info = load_grayscale(p)
    assert info["original_mode"] == "P"
    assert gray.shape == (30, 30)


# --------------------------------------------------------------------------
# detect_foreground
# --------------------------------------------------------------------------

def test_detect_foreground_finds_bilevel_ink():
    arr = np.full((60, 200), 255, dtype=np.uint8)
    arr[20:40, 50:150] = 0
    result = detect_foreground(arr)
    assert result["threshold"] == 0
    assert result["bbox"] == (50, 20, 149, 39)
    assert abs(result["fraction"] - (100 * 20) / (200 * 60)) < 1e-6  # exact: 2000/12000


def test_detect_foreground_blank_image_returns_none_bbox():
    arr = np.full((50, 50), 255, dtype=np.uint8)
    result = detect_foreground(arr)
    assert result["bbox"] is None
    assert result["fraction"] == 0.0


# --------------------------------------------------------------------------
# crop_with_margin
# --------------------------------------------------------------------------

def test_crop_with_margin_expands_correctly():
    arr = np.full((100, 100), 255, dtype=np.uint8)
    bbox = (40, 40, 59, 59)  # 20x20 box
    crop, crop_box, mx, my, clamped = crop_with_margin(arr, bbox, margin_frac=0.10)
    assert mx == 2 and my == 2  # 10% of 20 = 2
    assert crop_box == (38, 38, 61, 61)
    assert clamped is False


def test_crop_with_margin_clamps_at_image_edge():
    arr = np.full((50, 50), 255, dtype=np.uint8)
    bbox = (0, 0, 9, 9)  # touches top-left corner
    crop, crop_box, mx, my, clamped = crop_with_margin(arr, bbox, margin_frac=0.5)
    assert crop_box[0] == 0 and crop_box[1] == 0  # clamped, can't go negative
    assert clamped is True


# --------------------------------------------------------------------------
# resize_and_pad: aspect preservation, no distortion, padding behavior
# --------------------------------------------------------------------------

def test_resize_and_pad_preserves_aspect_ratio():
    crop = np.full((40, 100), 0, dtype=np.uint8)  # 100 wide x 40 tall, aspect 2.5
    canvas, info = resize_and_pad(crop, target_w=40, target_h=20, pad_value=255)
    assert canvas.shape == (20, 40)
    assert abs(info["aspect_distortion"]) < 1e-6  # never independently stretched


def test_resize_and_pad_output_dimensions_match_target():
    crop = np.full((10, 10), 0, dtype=np.uint8)
    canvas, info = resize_and_pad(crop, target_w=40, target_h=20, pad_value=255)
    assert canvas.shape == (20, 40)


def test_resize_and_pad_fills_padding_with_pad_value():
    crop = np.full((40, 40), 0, dtype=np.uint8)  # square content into a wide canvas
    canvas, info = resize_and_pad(crop, target_w=40, target_h=20, pad_value=255)
    # corners of a wide canvas holding square content should be pure padding
    assert canvas[0, 0] == 255
    assert canvas[-1, -1] == 255


def test_resize_and_pad_centers_content():
    # Square crop into a WIDE target -> padding only appears on the x-axis,
    # and it should be split evenly on both sides (centered), not flush left.
    crop = np.full((10, 10), 0, dtype=np.uint8)
    canvas, info = resize_and_pad(crop, target_w=40, target_h=20, pad_value=255)
    off_x, off_y = info["offset_x"], info["offset_y"]
    assert info["content_width"] < 40  # padding exists on the x-axis
    assert off_x == (40 - info["content_width"]) // 2
    assert off_x > 0  # not flush against the left edge
    assert off_y == 0  # content fills the full height, no y-axis padding expected here


# --------------------------------------------------------------------------
# Full pipeline: determinism, provenance, missing/corrupt handling, blank detection
# --------------------------------------------------------------------------

def test_preprocess_image_deterministic(tmp_path):
    p = tmp_path / "img.png"
    _make_bilevel_image(p)
    canvas1, rec1 = preprocess_image(p, TEST_CFG, margin_frac=0.05, intensity_variant="original")
    canvas2, rec2 = preprocess_image(p, TEST_CFG, margin_frac=0.05, intensity_variant="original")
    assert np.array_equal(canvas1, canvas2)
    assert rec1["output_sha256"] == rec2["output_sha256"]


def test_preprocess_image_missing_file_flagged_not_raised(tmp_path):
    canvas, rec = preprocess_image(tmp_path / "does_not_exist.png", TEST_CFG, 0.05, "original")
    assert canvas is None
    assert rec["preprocessing_status"] == "failed"
    assert rec["error"] == "file_not_found"


def test_preprocess_image_corrupt_file_flagged_not_raised(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"not a real png file")
    canvas, rec = preprocess_image(p, TEST_CFG, 0.05, "original")
    assert canvas is None
    assert rec["preprocessing_status"] == "failed"
    assert "unreadable" in rec["error"]


def test_preprocess_image_blank_input_flagged(tmp_path):
    p = tmp_path / "blank.png"
    Image.fromarray(np.full((60, 200), 255, dtype=np.uint8), mode="L").save(p)
    canvas, rec = preprocess_image(p, TEST_CFG, 0.05, "original")
    assert canvas is not None  # never dropped
    assert "no_foreground" in rec["flags"]
    assert "blank_output" in rec["flags"]
    assert rec["preprocessing_status"] == "flagged"


def test_preprocess_image_normal_case_status_ok(tmp_path):
    p = tmp_path / "img.png"
    _make_bilevel_image(p)
    canvas, rec = preprocess_image(p, TEST_CFG, margin_frac=0.05, intensity_variant="original")
    assert rec["preprocessing_status"] == "ok"
    assert rec["processed_width"] == 40
    assert rec["processed_height"] == 20
    assert rec["output_sha256"] is not None


def test_preprocess_image_provenance_fields_present(tmp_path):
    p = tmp_path / "img.png"
    _make_bilevel_image(p)
    canvas, rec = preprocess_image(p, TEST_CFG, margin_frac=0.05, intensity_variant="original")
    assert rec["original_width"] == 200
    assert rec["original_height"] == 60
    assert rec["preprocessing_version"] == "test-v1"
    assert rec["config_fingerprint"] is not None


def test_intensity_variant_flatten_bg_changes_output(tmp_path):
    p = tmp_path / "img.png"
    _make_grayscale_image(p, bg=235)  # continuous grayscale, background below 255
    canvas_orig, rec_orig = preprocess_image(p, TEST_CFG, 0.05, "original")
    canvas_flat, rec_flat = preprocess_image(p, TEST_CFG, 0.05, "flatten_bg")
    assert rec_flat["gain_applied"] > 1.0
    assert rec_orig["gain_applied"] == 1.0
    # flattened background should be brighter than the un-flattened one
    assert rec_flat["out_max"] >= rec_orig["out_max"]


def test_intensity_variant_flatten_bg_noop_on_already_white_background(tmp_path):
    p = tmp_path / "img.png"
    _make_bilevel_image(p)  # background already 255
    canvas, rec = preprocess_image(p, TEST_CFG, 0.05, "flatten_bg")
    assert abs(rec["gain_applied"] - 1.0) < 0.05  # near no-op


def test_margin_fraction_affects_crop_size(tmp_path):
    p = tmp_path / "img.png"
    _make_bilevel_image(p)
    _, rec_small = preprocess_image(p, TEST_CFG, margin_frac=0.05, intensity_variant="original")
    _, rec_large = preprocess_image(p, TEST_CFG, margin_frac=0.20, intensity_variant="original")
    assert rec_large["margin_px_x"] > rec_small["margin_px_x"]
    assert rec_large["margin_px_y"] > rec_small["margin_px_y"]


def test_extreme_aspect_ratio_flagged(tmp_path):
    p = tmp_path / "img.png"
    # a very thin, wide ink strip -> extreme aspect ratio
    arr = np.full((60, 300), 255, dtype=np.uint8)
    arr[28:30, 10:290] = 0  # 280 wide x 2 tall = aspect 140
    Image.fromarray(arr, mode="L").save(p)
    canvas, rec = preprocess_image(p, TEST_CFG, 0.05, "original")
    assert "extreme_aspect" in rec["flags"]


def test_rerun_consistency_same_record_fields(tmp_path):
    p = tmp_path / "img.png"
    _make_bilevel_image(p)
    _, rec1 = preprocess_image(p, TEST_CFG, 0.05, "original")
    _, rec2 = preprocess_image(p, TEST_CFG, 0.05, "original")
    assert rec1 == rec2
