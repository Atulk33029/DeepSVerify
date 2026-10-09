#!/usr/bin/env python3
"""
scripts/build_unified_dataset.py

End-to-end, re-runnable pipeline: inspect -> discover -> validate ->
hash/dup-detect -> unified metadata -> statistics -> validation report.

Usage:
    python scripts/build_unified_dataset.py --config configs/dataset_config.yaml

This script performs NO preprocessing, NO model code, NO training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dv_datasets.unified import (
    build_adapters,
    build_unified_metadata,
    write_metadata_csv,
    compute_statistics,
    detect_duplicates,
    write_duplicate_report,
    build_validation_report,
    run_quality_control,
)
from dv_datasets.common import UNKNOWN


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_inspection_report(adapters: dict) -> None:
    print("\n" + "=" * 60)
    print("DATASET INSPECTION (Section 6)")
    print("=" * 60)
    for name, adapter in adapters.items():
        result = adapter.inspect()
        d = result.to_dict()
        print(f"\nDataset: {d['dataset']}")
        print(f"  Total files:      {d['total_files']}")
        print(f"  Valid images:     {d['valid_images']}")
        print(f"  Invalid images:   {d['invalid_images']}")
        print(f"  Unparsed files:   {d['unparsed_files']}")
        print(f"  Writers:          {d['num_writers']}")
        print(f"  Genuine:          {d['genuine_count']}")
        print(f"  Forgery:          {d['forgery_count']}")
        print(f"  Unknown label:    {d['unknown_label_count']}")
        print(f"  File formats:     {d['extensions']}")
        print(f"  Common dims:      {d['common_dimensions'][:5]}")
        if d["invalid_files_sample"]:
            print(f"  Sample invalid:   {d['invalid_files_sample'][:3]}")
        if d["unparsed_paths_sample"]:
            print(f"  Sample unparsed:  {d['unparsed_paths_sample'][:3]}")


def maybe_copy_images(rows: list[dict], output_path: Path, copy_images: bool) -> None:
    if not copy_images:
        return
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if not r["is_valid"]:
            continue
        src = Path(r["original_path"])
        dst = images_dir / f'{r["image_id"]}{r["extension"]}'
        if dst.exists():
            continue  # already copied in a previous run; do not re-copy/corrupt
        shutil.copy2(src, dst)  # copy2 never modifies the source


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the DeepSVerify unified dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_config.yaml"))
    parser.add_argument("--skip-inspection-print", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_cfg = config.get("output", {})
    output_path = Path(output_cfg.get("output_path", "data/unified"))
    compute_phash = bool(output_cfg.get("compute_perceptual_hash", True))
    copy_images = bool(output_cfg.get("copy_images", False))

    print("Resolved paths:")
    print(f"  cedar.path          = {Path(config['cedar']['path']).resolve()}")
    print(f"  bhsig_bengali.path  = {Path(config['bhsig_bengali']['path']).resolve()}")
    print(f"  bhsig_hindi.path    = {Path(config['bhsig_hindi']['path']).resolve()}")
    print(f"  output.output_path  = {output_path.resolve()}")

    for key in ("cedar", "bhsig_bengali", "bhsig_hindi"):
        p = Path(config[key]["path"])
        if not p.exists():
            print(f"\nWARNING: configured path for '{key}' does not exist: {p.resolve()}")

    adapters = build_adapters(config)

    if not args.skip_inspection_print:
        print_inspection_report(adapters)

    print("\n" + "=" * 60)
    print("BUILDING UNIFIED METADATA")
    print("=" * 60)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=compute_phash)
    print(f"Total records discovered across all datasets: {len(rows)}")

    metadata_csv = output_path / "metadata.csv"
    write_metadata_csv(rows, metadata_csv)
    print(f"Wrote {metadata_csv}")

    stats = compute_statistics(rows)
    stats_json = output_path / "dataset_statistics.json"
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Wrote {stats_json}")

    dup_rows = detect_duplicates(rows)
    dup_csv = output_path / "duplicate_report.csv"
    write_duplicate_report(dup_rows, dup_csv)
    print(f"Wrote {dup_csv} ({len(dup_rows)} duplicate groups)")

    validation_report = build_validation_report(rows, dup_rows)
    validation_json = output_path / "validation_report.json"
    with open(validation_json, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, default=str)
    print(f"Wrote {validation_json}")

    qc = run_quality_control(rows, validation_report)

    maybe_copy_images(rows, output_path, copy_images)

    print("\n" + "=" * 60)
    print("FINAL SANITY CHECKS (Section 16)")
    print("=" * 60)
    for check, passed in qc.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")

    print("\n" + "=" * 60)
    print(f"VALIDATION STATUS: {validation_report['status']}")
    print("=" * 60)
    if validation_report["critical_failures"]:
        for cf in validation_report["critical_failures"]:
            print(f"  - {cf}")

    print("\nSUMMARY")
    print(f"  Total images:   {stats['overall']['total_images']}")
    print(f"  Valid images:   {stats['overall']['total_valid_images']}")
    print(f"  Invalid images: {stats['overall']['total_invalid_images']}")
    print(f"  Total writers:  {stats['overall']['total_writers']}")
    print(f"  Genuine:        {stats['overall']['total_genuine']}")
    print(f"  Forgery:        {stats['overall']['total_forgery']}")

    return 0 if validation_report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
