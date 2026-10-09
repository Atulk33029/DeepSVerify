"""
dv_datasets/bhsig_bengali.py

Adapter for the BHSig260 Bengali offline signature dataset.

ASSUMPTION (unverified against your actual downloaded copy -- confirm via
`inspect`): BHSig260 is commonly distributed as one subfolder per writer
(e.g. "001", "002", ...) directly under the dataset root, with filenames
of the form:

    B-S-<writer>-G-<sample>.tif   (genuine)
    B-S-<writer>-F-<sample>.tif   (forgery)

Writer-number subfolder and filename-embedded writer number are expected
to agree; if they don't, the writer_id embedded in the *filename* wins
and a parse_note records the mismatch (surfaced via inspection, not
silently resolved).

If a file's name does not match the configured genuine/forged regex, it
is still yielded (parse_ok=False) so it appears in the validation report
instead of being dropped.
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

DEFAULT_GENUINE_PATTERN = r"B-S-(?P<writer>\d+)-G-(?P<sample>\d+)\.(?P<ext>\w+)$"
DEFAULT_FORGED_PATTERN = r"B-S-(?P<writer>\d+)-F-(?P<sample>\d+)\.(?P<ext>\w+)$"
LANGUAGE = "Bengali"


class BHSigBengaliAdapter(DatasetAdapter):
    dataset_name = DatasetName.BHSIG_BENGALI

    def _patterns(self) -> dict:
        return {
            SignatureType.GENUINE: self.config.get("genuine_pattern", DEFAULT_GENUINE_PATTERN),
            SignatureType.FORGERY: self.config.get("forged_pattern", DEFAULT_FORGED_PATTERN),
        }

    def _layout(self) -> str:
        return self.config.get("layout", "legacy")

    def _candidate_files(self) -> Iterable[Path]:
        files = [
            p for p in self.root_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(files, key=lambda x: str(x))

    def discover(self) -> Iterable[RawRecord]:
        patterns = self._patterns()

        if self._layout() == "writer_folders":
            for path in self._candidate_files():
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
                    language=LANGUAGE,
                    parse_ok=(sig_type in SignatureType.ALLOWED),
                    parse_note=note,
                )
            return

        for path in self._candidate_files():
            sig_type, writer, _ = parse_with_patterns(path.name, patterns)
            folder_writer = path.parent.name if path.parent != self.root_path else None

            if sig_type is None:
                yield RawRecord(
                    dataset=self.dataset_name,
                    writer_id=folder_writer or UNKNOWN,
                    signature_type=UNKNOWN,
                    original_path=path,
                    language=LANGUAGE,
                    parse_ok=False,
                    parse_note="filename did not match configured BHSig260 Bengali patterns",
                )
                continue

            note = ""
            if folder_writer and folder_writer.lstrip("0") != writer.lstrip("0") and folder_writer != writer:
                note = f"writer_id mismatch: filename={writer} folder={folder_writer}; filename value used"

            yield RawRecord(
                dataset=self.dataset_name,
                writer_id=writer,
                signature_type=sig_type,
                original_path=path,
                language=LANGUAGE,
                parse_ok=True,
                parse_note=note,
            )
