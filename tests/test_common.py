from pathlib import Path

from dv_datasets.common import (
    validate_image,
    sha256_file,
    make_image_id,
    make_global_writer_id,
    parse_with_patterns,
    SignatureType,
    DatasetName,
)


def test_validate_image_valid(cedar_root):
    p = cedar_root / "full_org" / "original_1_1.png"
    v = validate_image(p)
    assert v.valid is True
    assert v.width == 32
    assert v.height == 20


def test_validate_image_missing(tmp_path):
    v = validate_image(tmp_path / "does_not_exist.png")
    assert v.valid is False
    assert v.error == "file_not_found"


def test_validate_image_corrupt(corrupt_image):
    v = validate_image(corrupt_image)
    assert v.valid is False
    assert v.error is not None


def test_sha256_deterministic(cedar_root):
    p = cedar_root / "full_org" / "original_1_1.png"
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_make_image_id_format():
    assert make_image_id(DatasetName.CEDAR, 1) == "CEDAR_000001"
    assert make_image_id(DatasetName.BHSIG_BENGALI, 42) == "BHSIGB_000042"
    assert make_image_id(DatasetName.BHSIG_HINDI, 7) == "BHSIGH_000007"


def test_make_global_writer_id_no_collision():
    a = make_global_writer_id(DatasetName.CEDAR, "001")
    b = make_global_writer_id(DatasetName.BHSIG_HINDI, "001")
    assert a != b
    assert a == "CEDAR_001"
    assert b == "BHSIG_HINDI_001"


def test_parse_with_patterns_match():
    patterns = {
        SignatureType.GENUINE: r"original_(?P<writer>\d+)_(?P<sample>\d+)\.(?P<ext>\w+)$",
    }
    sig, writer, m = parse_with_patterns("original_5_2.png", patterns)
    assert sig == SignatureType.GENUINE
    assert writer == "5"


def test_parse_with_patterns_no_match():
    patterns = {
        SignatureType.GENUINE: r"original_(?P<writer>\d+)_(?P<sample>\d+)\.(?P<ext>\w+)$",
    }
    sig, writer, m = parse_with_patterns("totally_unrelated.png", patterns)
    assert sig is None
    assert writer is None
