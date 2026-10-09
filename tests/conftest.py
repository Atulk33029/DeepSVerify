"""
tests/conftest.py

Builds tiny SYNTHETIC dataset trees (a handful of generated PNGs, not real
signatures) that mimic the assumed CEDAR / BHSig260 layouts, so the
pipeline can be tested end-to-end without access to the real, large
source datasets. This also exercises the adapters' filename-parsing
logic against the documented assumptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


def _make_png(path: Path, size=(32, 20), color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


@pytest.fixture
def cedar_root(tmp_path) -> Path:
    root = tmp_path / "cedar"
    org = root / "full_org"
    forg = root / "full_forg"
    for writer in (1, 2):
        for sample in (1, 2):
            _make_png(org / f"original_{writer}_{sample}.png")
            _make_png(forg / f"forgeries_{writer}_{sample}.png")
    # one deliberately unparsable file
    _make_png(org / "weird_name_file.png")
    return root


@pytest.fixture
def bhsig_bengali_root(tmp_path) -> Path:
    root = tmp_path / "bhsig_bengali"
    for writer in ("001", "002"):
        for sample in ("01", "02"):
            _make_png(root / writer / f"B-S-{writer}-G-{sample}.tif")
            _make_png(root / writer / f"B-S-{writer}-F-{sample}.tif")
    return root


@pytest.fixture
def bhsig_hindi_root(tmp_path) -> Path:
    root = tmp_path / "bhsig_hindi"
    for writer in ("001", "002"):
        for sample in ("01", "02"):
            _make_png(root / writer / f"H-S-{writer}-G-{sample}.tif")
            _make_png(root / writer / f"H-S-{writer}-F-{sample}.tif")
    return root


@pytest.fixture
def writer_folders_root(tmp_path) -> Path:
    """Mimics the combined-Kaggle-mirror layout: <root>/<writer>/images/<file>,
    with genuine/forgery distinguished by an explicit subfolder name rather
    than a filename convention -- exercising the generic keyword fallback
    classifier rather than the configured regex patterns.
    """
    root = tmp_path / "writer_folders_ds"
    for writer in ("1", "2"):
        genuine_dir = root / writer / "images" / "genuine"
        forged_dir = root / writer / "images" / "forged"
        for sample in (1, 2):
            _make_png(genuine_dir / f"{sample}.png")
            _make_png(forged_dir / f"{sample}.png")
    # one file with no genuine/forgery marker anywhere -> should stay UNKNOWN
    _make_png(root / "3" / "images" / "mystery.png")
    return root


@pytest.fixture
def corrupt_image(tmp_path) -> Path:
    p = tmp_path / "corrupt" / "original_9_9.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"this is not a valid png file")
    return p
