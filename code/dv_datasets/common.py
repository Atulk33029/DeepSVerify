"""
dv_datasets/common.py

Shared constants, utilities, and the abstract adapter interface used by
every dataset-specific adapter (cedar.py, bhsig_bengali.py, bhsig_hindi.py).

Design notes
------------
- This stage performs NO preprocessing (no resize/threshold/normalize).
- All operations are deterministic: no randomness, stable sort order,
  stable ID generation based on sorted (writer_id, sample identifiers).
- Filename parsing is done via *configurable* regex patterns (see
  configs/dataset_config.yaml) rather than hard-coded string logic, so
  that if the real downloaded datasets use a slightly different naming
  convention than the commonly published one, the pipeline can be
  adapted without touching code. Any file that fails to match its
  dataset's configured pattern(s) is NOT silently dropped -- it is
  recorded as an "unparsed" file and surfaced in the validation report.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, UnidentifiedImageError

try:
    import imagehash  # optional, only needed if compute_perceptual_hash=true
    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False


# --------------------------------------------------------------------------
# Terminology constants (Section 3 of the spec) -- use these everywhere,
# never free-form strings, so typos can't silently create a new label.
# --------------------------------------------------------------------------

class SignatureType:
    GENUINE = "genuine"
    FORGERY = "forgery"
    ALLOWED = {GENUINE, FORGERY}


class DatasetName:
    CEDAR = "CEDAR"
    BHSIG_BENGALI = "BHSIG_BENGALI"
    BHSIG_HINDI = "BHSIG_HINDI"
    ALLOWED = {CEDAR, BHSIG_BENGALI, BHSIG_HINDI}


# Prefix used for the unified, globally-unique image_id (Section 9).
IMAGE_ID_PREFIX = {
    DatasetName.CEDAR: "CEDAR",
    DatasetName.BHSIG_BENGALI: "BHSIGB",
    DatasetName.BHSIG_HINDI: "BHSIGH",
}

UNKNOWN = "unknown"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


# --------------------------------------------------------------------------
# Raw record produced by each adapter, BEFORE global_writer_id / image_id
# are assigned by dv_datasets/unified.py (those require cross-dataset context).
# --------------------------------------------------------------------------

@dataclass
class RawRecord:
    dataset: str
    writer_id: str
    signature_type: str          # SignatureType.GENUINE / FORGERY / UNKNOWN
    original_path: Path
    language: str = UNKNOWN
    parse_ok: bool = True
    parse_note: str = ""


@dataclass
class InspectionResult:
    dataset: str
    total_files: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    unparsed_files: int = 0
    writers: set = field(default_factory=set)
    genuine_count: int = 0
    forgery_count: int = 0
    unknown_label_count: int = 0
    extensions: dict = field(default_factory=dict)      # ext -> count
    dimensions: dict = field(default_factory=dict)       # (w,h) -> count
    color_modes: dict = field(default_factory=dict)      # mode -> count
    invalid_files: list = field(default_factory=list)    # (path, reason)
    unparsed_paths: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "total_files": self.total_files,
            "valid_images": self.valid_images,
            "invalid_images": self.invalid_images,
            "unparsed_files": self.unparsed_files,
            "num_writers": len(self.writers),
            "genuine_count": self.genuine_count,
            "forgery_count": self.forgery_count,
            "unknown_label_count": self.unknown_label_count,
            "extensions": self.extensions,
            "common_dimensions": sorted(
                self.dimensions.items(), key=lambda kv: -kv[1]
            )[:10],
            "color_modes": self.color_modes,
            "invalid_files_sample": self.invalid_files[:25],
            "unparsed_paths_sample": [str(p) for p in self.unparsed_paths[:25]],
        }


# --------------------------------------------------------------------------
# Image validation (Section 7) -- validation only, no modification.
# --------------------------------------------------------------------------

@dataclass
class ImageValidation:
    valid: bool
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    format: Optional[str] = None
    error: Optional[str] = None


def validate_image(path: Path) -> ImageValidation:
    """Open (without loading full pixel data twice) and sanity-check an image.

    Checks: existence, readability, valid encoding, non-zero dimensions,
    supported extension. Never resizes/converts/writes anything.
    """
    if not path.exists():
        return ImageValidation(valid=False, error="file_not_found")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return ImageValidation(valid=False, error=f"unsupported_extension:{path.suffix}")

    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check
        # verify() invalidates the file handle for further use, so reopen
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            fmt = img.format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageValidation(valid=False, error=f"unreadable:{exc}")

    if width <= 0 or height <= 0:
        return ImageValidation(valid=False, error="zero_or_negative_dimension")

    return ImageValidation(valid=True, width=width, height=height, mode=mode, format=fmt)


# --------------------------------------------------------------------------
# Hashing (Section 8: duplicate detection)
# --------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path) -> Optional[str]:
    """Return a perceptual hash (phash) string, or None if imagehash is
    unavailable or the image cannot be hashed. Never raises.
    """
    if not _HAS_IMAGEHASH:
        return None
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Deterministic ID generation (Section 9)
# --------------------------------------------------------------------------

def make_image_id(dataset: str, sequence_number: int) -> str:
    prefix = IMAGE_ID_PREFIX[dataset]
    return f"{prefix}_{sequence_number:06d}"


def make_global_writer_id(dataset: str, writer_id: str) -> str:
    """Namespaces a writer_id by dataset so writer '001' in CEDAR and
    writer '001' in BHSig260 Hindi can never collide (Section 3).
    """
    return f"{dataset}_{writer_id}"


# --------------------------------------------------------------------------
# Regex-driven filename parsing helper shared by adapters.
# --------------------------------------------------------------------------

def parse_with_patterns(filename: str, patterns: dict[str, str]) -> tuple[Optional[str], Optional[str], Optional[re.Match]]:
    """Try each (signature_type -> regex) pattern against filename.

    Returns (signature_type, writer_id, match) for the first pattern that
    matches, or (None, None, None) if nothing matches. Patterns must define
    a named group 'writer'.
    """
    for sig_type, pattern in patterns.items():
        m = re.match(pattern, filename)
        if m:
            groupdict = m.groupdict()
            writer = groupdict.get("writer")
            return sig_type, writer, m
    return None, None, None


# --------------------------------------------------------------------------
# "writer_folders" layout support.
#
# Several real-world redistributions of CEDAR / BHSig260 (e.g. combined
# Kaggle mirrors) organize files as:
#
#     <dataset_root>/<writer_folder>/images/<file>
#     <dataset_root>/<writer_folder>/<file>
#
# rather than the "legacy" full_org/full_forg or flat B-S-<writer>-G-<n>
# filename convention this pipeline originally assumed. Since the exact
# filenames *inside* each writer folder vary across mirrors and were not
# verified against your specific download, this layout is handled with a
# best-effort, fully-reported classifier rather than a hard-coded regex:
#
#   1. try the dataset's configured genuine_pattern/forged_pattern regex
#      against the filename first (works if the mirror does use one of
#      the known conventions inside each writer folder)
#   2. else, look for genuine/forgery keywords in the path's directory
#      components or filename (e.g. "genuine"/"forged" subfolders, or
#      "-G-"/"-F-" tokens)
#   3. else, leave it UNKNOWN / parse_ok=False so it surfaces in the
#      validation report instead of being silently guessed.
#
# writer_id is taken from the top-level folder name under the dataset
# root (the "<writer_folder>" component), which is far more reliable
# here than trying to parse a writer number back out of the filename.
# --------------------------------------------------------------------------

GENERIC_GENUINE_KEYWORDS = ["genuine", "original", "real", "-g-", "_g_", ".g."]
GENERIC_FORGERY_KEYWORDS = ["forg", "fake", "skilled", "disguise", "-f-", "_f_", ".f."]


def writer_id_from_top_folder(path: Path, root: Path) -> str:
    """Returns the first path component under root, e.g. for
    root/42/images/foo.png with root=<root>, returns '42'.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return UNKNOWN
    return rel.parts[0] if rel.parts else UNKNOWN


def classify_generic(path: Path, root: Path) -> tuple[str, str]:
    """Best-effort genuine/forgery classification for the writer_folders
    layout, used only as a fallback after configured regex patterns have
    already been tried and failed. Never raises; returns (UNKNOWN, note)
    if nothing matches so the caller can mark parse_ok=False.
    """
    try:
        rel_parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    except ValueError:
        rel_parts = []
    name_lower = path.name.lower()
    haystacks = rel_parts + [name_lower]

    # Check forgery keywords first: "original" contains no forgery
    # substrings, but some genuine keywords could theoretically overlap
    # with forgery ones in unusual mirrors, so order is deliberate and
    # documented rather than incidental.
    for h in haystacks:
        for kw in GENERIC_FORGERY_KEYWORDS:
            if kw in h:
                return SignatureType.FORGERY, f"generic classifier: matched forgery keyword '{kw}' in '{h}'"
    for h in haystacks:
        for kw in GENERIC_GENUINE_KEYWORDS:
            if kw in h:
                return SignatureType.GENUINE, f"generic classifier: matched genuine keyword '{kw}' in '{h}'"

    return UNKNOWN, "generic classifier: no genuine/forgery marker found in path or filename"


# --------------------------------------------------------------------------
# Abstract adapter interface (Section 5)
# --------------------------------------------------------------------------

class DatasetAdapter(ABC):
    """Every dataset adapter understands ONLY its own source layout and
    exposes this common interface to the unified builder.
    """

    dataset_name: str

    def __init__(self, root_path: Path, config: dict):
        self.root_path = Path(root_path)
        self.config = config

    @abstractmethod
    def discover(self) -> Iterable[RawRecord]:
        """Walk the source directory and yield RawRecord for every file
        found with a supported image extension (parsed or not -- unparsed
        files are yielded with parse_ok=False so they surface in reports
        rather than vanishing silently).
        """
        raise NotImplementedError

    def inspect(self) -> InspectionResult:
        """Non-destructive inspection pass (Section 6). Re-walks discover()
        and validates every image, without writing anything.
        """
        result = InspectionResult(dataset=self.dataset_name)
        for rec in self.discover():
            result.total_files += 1
            ext = rec.original_path.suffix.lower()
            result.extensions[ext] = result.extensions.get(ext, 0) + 1

            if not rec.parse_ok:
                result.unparsed_files += 1
                result.unparsed_paths.append(rec.original_path)

            validation = validate_image(rec.original_path)
            if validation.valid:
                result.valid_images += 1
                dim_key = f"{validation.width}x{validation.height}"
                result.dimensions[dim_key] = result.dimensions.get(dim_key, 0) + 1
                result.color_modes[validation.mode] = result.color_modes.get(validation.mode, 0) + 1
            else:
                result.invalid_images += 1
                result.invalid_files.append((str(rec.original_path), validation.error))

            if rec.writer_id and rec.writer_id != UNKNOWN:
                result.writers.add(rec.writer_id)

            if rec.signature_type == SignatureType.GENUINE:
                result.genuine_count += 1
            elif rec.signature_type == SignatureType.FORGERY:
                result.forgery_count += 1
            else:
                result.unknown_label_count += 1

        return result
