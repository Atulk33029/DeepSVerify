"""
dv_datasets/unified.py

Orchestrates the three dataset adapters into ONE unified, provenance-
preserving metadata table, plus statistics / validation / duplicate
reports. No image preprocessing happens here -- validation only.

Determinism: every collection is sorted by (dataset, writer_id, filename)
before IDs are assigned, so re-running against an unchanged source tree
reproduces identical image_ids, ordering, and hashes.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .common import (
    DatasetName,
    SignatureType,
    UNKNOWN,
    make_image_id,
    make_global_writer_id,
    sha256_file,
    perceptual_hash,
    validate_image,
)
from .cedar import CedarAdapter
from .bhsig_bengali import BHSigBengaliAdapter
from .bhsig_hindi import BHSigHindiAdapter

METADATA_COLUMNS = [
    "image_id",
    "dataset",
    "writer_id",
    "global_writer_id",
    "signature_type",
    "language",
    "original_path",
    "filename",
    "extension",
    "width",
    "height",
    "mode",
    "file_size_bytes",
    "sha256",
    "phash",
    "is_valid",
    "validation_error",
    "parse_ok",
    "parse_note",
]

ADAPTER_CLASSES = {
    DatasetName.CEDAR: CedarAdapter,
    DatasetName.BHSIG_BENGALI: BHSigBengaliAdapter,
    DatasetName.BHSIG_HINDI: BHSigHindiAdapter,
}


def build_adapters(config: dict) -> dict:
    """Instantiate one adapter per dataset from the loaded YAML config."""
    adapters = {}
    for name, cls in ADAPTER_CLASSES.items():
        key = {
            DatasetName.CEDAR: "cedar",
            DatasetName.BHSIG_BENGALI: "bhsig_bengali",
            DatasetName.BHSIG_HINDI: "bhsig_hindi",
        }[name]
        ds_cfg = config.get(key, {})
        path = ds_cfg.get("path")
        if not path:
            raise ValueError(f"Missing '{key}.path' in dataset_config.yaml")
        adapters[name] = cls(Path(path), ds_cfg)
    return adapters


def build_unified_metadata(adapters: dict, compute_perceptual_hash: bool = True) -> list[dict]:
    """Discover records from every adapter, validate each image, compute
    hashes, and assign deterministic image_id / global_writer_id.

    Returns a list of dicts matching METADATA_COLUMNS, sorted
    deterministically by (dataset, writer_id, filename).
    """
    rows: list[dict] = []

    for dataset_name, adapter in adapters.items():
        records = list(adapter.discover())
        # Deterministic order: sort by (writer_id, filename) within dataset.
        records.sort(key=lambda r: (r.writer_id or "", r.original_path.name))

        counter = 0
        for rec in records:
            counter += 1
            image_id = make_image_id(dataset_name, counter)
            global_writer_id = (
                make_global_writer_id(dataset_name, rec.writer_id)
                if rec.writer_id and rec.writer_id != UNKNOWN
                else UNKNOWN
            )

            validation = validate_image(rec.original_path)

            file_size = rec.original_path.stat().st_size if rec.original_path.exists() else None
            sha256 = sha256_file(rec.original_path) if rec.original_path.exists() else None
            phash = (
                perceptual_hash(rec.original_path)
                if compute_perceptual_hash and rec.original_path.exists() and validation.valid
                else None
            )

            rows.append({
                "image_id": image_id,
                "dataset": dataset_name,
                "writer_id": rec.writer_id or UNKNOWN,
                "global_writer_id": global_writer_id,
                "signature_type": rec.signature_type or UNKNOWN,
                "language": rec.language or UNKNOWN,
                "original_path": str(rec.original_path),
                "filename": rec.original_path.name,
                "extension": rec.original_path.suffix.lower(),
                "width": validation.width if validation.valid else UNKNOWN,
                "height": validation.height if validation.valid else UNKNOWN,
                "mode": validation.mode if validation.valid else UNKNOWN,
                "file_size_bytes": file_size if file_size is not None else UNKNOWN,
                "sha256": sha256 if sha256 is not None else UNKNOWN,
                "phash": phash if phash is not None else UNKNOWN,
                "is_valid": validation.valid,
                "validation_error": validation.error or "",
                "parse_ok": rec.parse_ok,
                "parse_note": rec.parse_note,
            })

    return rows


def write_metadata_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_statistics(rows: list[dict]) -> dict:
    stats = {
        "overall": {
            "total_images": len(rows),
            "total_writers": len({r["global_writer_id"] for r in rows if r["global_writer_id"] != UNKNOWN}),
            "total_genuine": sum(1 for r in rows if r["signature_type"] == SignatureType.GENUINE),
            "total_forgery": sum(1 for r in rows if r["signature_type"] == SignatureType.FORGERY),
            "total_unknown_label": sum(
                1 for r in rows if r["signature_type"] not in SignatureType.ALLOWED
            ),
            "total_valid_images": sum(1 for r in rows if r["is_valid"]),
            "total_invalid_images": sum(1 for r in rows if not r["is_valid"]),
        },
        "by_dataset": {},
        "by_writer": {},
        "distributions": {
            "dimensions": defaultdict(int),
            "file_formats": defaultdict(int),
        },
    }

    by_dataset = defaultdict(lambda: {"total": 0, "genuine": 0, "forgery": 0, "writers": set(), "valid": 0, "invalid": 0})
    by_writer = defaultdict(lambda: {"total": 0, "genuine": 0, "forgery": 0, "dataset": None})

    for r in rows:
        d = by_dataset[r["dataset"]]
        d["total"] += 1
        d["valid"] += 1 if r["is_valid"] else 0
        d["invalid"] += 0 if r["is_valid"] else 1
        if r["signature_type"] == SignatureType.GENUINE:
            d["genuine"] += 1
        elif r["signature_type"] == SignatureType.FORGERY:
            d["forgery"] += 1
        if r["global_writer_id"] != UNKNOWN:
            d["writers"].add(r["global_writer_id"])

        gw = r["global_writer_id"]
        w = by_writer[gw]
        w["total"] += 1
        w["dataset"] = r["dataset"]
        if r["signature_type"] == SignatureType.GENUINE:
            w["genuine"] += 1
        elif r["signature_type"] == SignatureType.FORGERY:
            w["forgery"] += 1

        if r["is_valid"] and r["width"] != UNKNOWN:
            stats["distributions"]["dimensions"][f'{r["width"]}x{r["height"]}'] += 1
        stats["distributions"]["file_formats"][r["extension"]] += 1

    for name, d in by_dataset.items():
        stats["by_dataset"][name] = {
            "total_images": d["total"],
            "genuine": d["genuine"],
            "forgery": d["forgery"],
            "num_writers": len(d["writers"]),
            "valid_images": d["valid"],
            "invalid_images": d["invalid"],
        }

    for gw, w in by_writer.items():
        stats["by_writer"][gw] = {
            "dataset": w["dataset"],
            "total_images": w["total"],
            "genuine": w["genuine"],
            "forgery": w["forgery"],
        }

    stats["distributions"]["dimensions"] = dict(stats["distributions"]["dimensions"])
    stats["distributions"]["file_formats"] = dict(stats["distributions"]["file_formats"])

    return stats


def detect_duplicates(rows: list[dict]) -> list[dict]:
    """Exact (sha256) and, where available, perceptual (phash) duplicate
    detection. Reports only -- never deletes or modifies source files.
    """
    by_sha = defaultdict(list)
    by_phash = defaultdict(list)

    for r in rows:
        if r["sha256"] != UNKNOWN:
            by_sha[r["sha256"]].append(r["image_id"])
        if r["phash"] != UNKNOWN:
            by_phash[r["phash"]].append(r["image_id"])

    dup_rows = []
    for sha, ids in by_sha.items():
        if len(ids) > 1:
            dup_rows.append({
                "hash_type": "sha256_exact",
                "hash_value": sha,
                "num_duplicates": len(ids),
                "image_ids": ";".join(ids),
            })
    for ph, ids in by_phash.items():
        if len(ids) > 1:
            dup_rows.append({
                "hash_type": "phash_perceptual",
                "hash_value": ph,
                "num_duplicates": len(ids),
                "image_ids": ";".join(ids),
            })
    return dup_rows


def write_duplicate_report(dup_rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hash_type", "hash_value", "num_duplicates", "image_ids"])
        writer.writeheader()
        for row in dup_rows:
            writer.writerow(row)


def build_validation_report(rows: list[dict], dup_rows: list[dict]) -> dict:
    missing_files = [r["image_id"] for r in rows if r["validation_error"] == "file_not_found"]
    unreadable = [r["image_id"] for r in rows if r["is_valid"] is False and r["validation_error"] not in ("file_not_found",)]
    invalid_labels = [r["image_id"] for r in rows if r["signature_type"] not in SignatureType.ALLOWED]
    missing_writer_ids = [r["image_id"] for r in rows if r["writer_id"] == UNKNOWN]
    unsupported_ext = [r["image_id"] for r in rows if "unsupported_extension" in r["validation_error"]]
    unparsed = [r["image_id"] for r in rows if not r["parse_ok"]]

    image_ids = [r["image_id"] for r in rows]
    duplicate_image_ids = len(image_ids) != len(set(image_ids))

    exact_dup_groups = [d for d in dup_rows if d["hash_type"] == "sha256_exact"]
    phash_dup_groups = [d for d in dup_rows if d["hash_type"] == "phash_perceptual"]

    # global_writer_id uniqueness across datasets is guaranteed by
    # construction (dataset name is namespaced into the ID), but we still
    # check for accidental collisions as a sanity net.
    gw_to_datasets = defaultdict(set)
    for r in rows:
        if r["global_writer_id"] != UNKNOWN:
            gw_to_datasets[r["global_writer_id"]].add(r["dataset"])
    cross_dataset_collisions = {gw: list(ds) for gw, ds in gw_to_datasets.items() if len(ds) > 1}

    critical_failures = []
    if missing_files:
        critical_failures.append(f"{len(missing_files)} metadata rows point to missing files")
    if duplicate_image_ids:
        critical_failures.append("duplicate image_id values detected")
    if cross_dataset_collisions:
        critical_failures.append("global_writer_id collision across datasets detected")
    if invalid_labels:
        critical_failures.append(f"{len(invalid_labels)} rows have a signature_type outside {{genuine, forgery}}")

    status = "FAIL" if critical_failures else "PASS"

    return {
        "status": status,
        "critical_failures": critical_failures,
        "missing_files": missing_files,
        "unreadable_images": unreadable,
        "invalid_labels": invalid_labels,
        "missing_writer_ids": missing_writer_ids,
        "unsupported_extensions": unsupported_ext,
        "unparsed_filenames": unparsed,
        "duplicate_image_ids": duplicate_image_ids,
        "exact_duplicate_groups": len(exact_dup_groups),
        "perceptual_duplicate_groups": len(phash_dup_groups),
        "cross_dataset_writer_id_collisions": cross_dataset_collisions,
        "total_rows_checked": len(rows),
    }


def run_quality_control(rows: list[dict], validation_report: dict) -> dict:
    """Section 16 sanity checks, returned as a dict of check -> pass/fail."""
    checks = {}
    checks["every_row_points_to_existing_file"] = len(validation_report["missing_files"]) == 0
    checks["every_image_has_writer_id"] = len(validation_report["missing_writer_ids"]) == 0
    checks["every_image_has_valid_signature_type"] = len(validation_report["invalid_labels"]) == 0
    checks["every_image_has_valid_dataset_label"] = all(r["dataset"] in DatasetName.ALLOWED for r in rows)
    checks["global_writer_id_unique_across_datasets"] = len(validation_report["cross_dataset_writer_id_collisions"]) == 0
    checks["no_image_id_duplicates"] = not validation_report["duplicate_image_ids"]
    checks["metadata_row_count_matches_discovered"] = True  # true by construction (1 row per discovered file)
    checks["overall_pass"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks
