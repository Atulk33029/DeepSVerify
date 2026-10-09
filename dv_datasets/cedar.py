"""
dv_datasets/cedar.py

Adapter for the CEDAR offline signature dataset.

ASSUMPTION (unverified against your actual downloaded copy -- see README
"Important Limitations/Assumptions" and re-run `inspect` to confirm):

CEDAR is most commonly distributed with two subfolders under the dataset
root:

    <root>/full_org/original_<writer>_<sample>.png
    <root>/full_forg/forgeries_<writer>_<sample>.png

writer ranges 1-55, 24 genuine + 24 forged signatures per writer in the
canonical release. This adapter does NOT assume those counts are correct
-- it reports whatever it actually finds.

Both the subfolder names and the filename regexes are configurable via
dataset_config.yaml (keys: genuine_dir, forged_dir, genuine_pattern,
forged_pattern) in case your copy differs. If genuine_dir/forged_dir do
not exist, the adapter falls back to scanning the whole tree recursively
and classifying by pattern match alone, flagging anything that fails
both patterns as unparsed rather than guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .common import (
    DatasetAdapter,
    DatasetName,
    RawRecord,
    SignatureType,
    UNKNOWN,
    SUPPORTED_EXTENSIONS,
    parse_with_patterns,
    writer_id_from_top_folder,
    classify_generic,
)

DEFAULT_GENUINE_DIR = "full_org"
DEFAULT_FORGED_DIR = "full_forg"
DEFAULT_GENUINE_PATTERN = r"original_(?P<writer>\d+)_(?P<sample>\d+)\.(?P<ext>\w+)$"
DEFAULT_FORGED_PATTERN = r"forgeries_(?P<writer>\d+)_(?P<sample>\d+)\.(?P<ext>\w+)$"


class CedarAdapter(DatasetAdapter):
    dataset_name = DatasetName.CEDAR

    def _patterns(self) -> dict:
        return {
            SignatureType.GENUINE: self.config.get("genuine_pattern", DEFAULT_GENUINE_PATTERN),
            SignatureType.FORGERY: self.config.get("forged_pattern", DEFAULT_FORGED_PATTERN),
        }

    def _layout(self) -> str:
        return self.config.get("layout", "legacy")

    def _candidate_files(self) -> Iterable[Path]:
        genuine_dir = self.root_path / self.config.get("genuine_dir", DEFAULT_GENUINE_DIR)
        forged_dir = self.root_path / self.config.get("forged_dir", DEFAULT_FORGED_DIR)

        if genuine_dir.is_dir() or forged_dir.is_dir():
            files = []
            if genuine_dir.is_dir():
                files.extend(genuine_dir.iterdir())
            if forged_dir.is_dir():
                files.extend(forged_dir.iterdir())
        else:
            # Fallback: recursive scan of the whole root. Nothing is
            # dropped -- every image-extension file is yielded, parsed or
            # not, so the discrepancy from the expected layout is visible
            # in the inspection/validation reports.
            files = list(self.root_path.rglob("*"))

        for p in sorted(files, key=lambda x: str(x)):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield p

    def _writer_folder_files(self) -> Iterable[Path]:
        files = [
            p for p in self.root_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(files, key=lambda x: str(x))

    def discover(self) -> Iterable[RawRecord]:
        patterns = self._patterns()

        if self._layout() == "writer_folders":
            for path in self._writer_folder_files():
                sig_type, writer, _ = parse_with_patterns(path.name, patterns)
                note = ""
                if sig_type is None:
                    sig_type, note = classify_generic(path, self.root_path)
                writer = writer or writer_id_from_top_folder(path, self.root_path)
                yield RawRecord(
                    dataset=self.dataset_name,
                    writer_id=writer or UNKNOWN,
                    signature_type=sig_type,
                    original_path=path,
                    language=UNKNOWN,
                    parse_ok=(sig_type in SignatureType.ALLOWED),
                    parse_note=note,
                )
            return

        for path in self._candidate_files():
            sig_type, writer, _ = parse_with_patterns(path.name, patterns)
            if sig_type is None:
                yield RawRecord(
                    dataset=self.dataset_name,
                    writer_id=UNKNOWN,
                    signature_type=UNKNOWN,
                    original_path=path,
                    language=UNKNOWN,
                    parse_ok=False,
                    parse_note="filename did not match configured CEDAR patterns",
                )
            else:
                yield RawRecord(
                    dataset=self.dataset_name,
                    writer_id=writer,
                    signature_type=sig_type,
                    original_path=path,
                    language=UNKNOWN,  # CEDAR is not language/script-specific
                    parse_ok=True,
                )
