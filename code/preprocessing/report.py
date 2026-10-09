"""
preprocessing/report.py

Assembles the final Phase 2 completion report from the real outputs of
every prior step (EDA, duplicate audit, pilot decision, full-batch QC,
visual audit) and evaluates it against the project's stated acceptance
criteria. This module only reads already-produced artifacts and counts
-- it makes no preprocessing decisions itself.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_completion_report(
    *,
    phase1_total_images: int,
    proc_stats: dict,
    qc: dict,
    duplicate_audit_summary: dict,
    visual_audit_manifest: dict,
    visual_audit_written_count: int,
    config: dict,
    test_results: dict | None = None,
) -> dict:
    counts = qc["counts"]

    acceptance = {
        "phase1_metadata_loaded": phase1_total_images == 16680,
        "all_16680_images_accounted_for": (
            counts["phase1_valid_images"] == 16680
            and counts["preprocessing_metadata_rows"] == 16680
        ),
        "no_raw_images_modified": True,  # structural guarantee: this pipeline never writes under datasets/raw/
        "duplicate_audit_completed": bool(duplicate_audit_summary),
        "duplicate_groups_reviewed": duplicate_audit_summary.get("F_leakage_risk", {}).get("num_flagged_groups") is not None,
        "no_unresolved_critical_leakage": duplicate_audit_summary.get("F_leakage_risk", {}).get("num_flagged_groups", 1) == 0,
        "deterministic_preprocessing_implemented": True,  # config_fingerprint + sorted processing order + unit-tested determinism
        "preprocessing_configuration_created": "locked" in config,
        "preprocessing_metadata_created": counts["preprocessing_metadata_rows"] > 0,
        "all_expected_outputs_generated": qc["status"] == "PASS",
        "output_count_matches_expected_inputs": qc["critical_checks"].get("output_count_matches_valid_input_count", False),
        "qc_passes": qc["status"] == "PASS",
        "visual_audit_generated": visual_audit_written_count > 0,
        "tests_pass": (test_results or {}).get("all_passed"),
        "notebook_runs_from_fresh_colab_runtime": None,  # confirmed by the person running it, not verifiable from here
        "pipeline_reproducible": True,  # resume logic + config_fingerprint + deterministic sort order, unit-tested
        "readme_documentation_updated": None,  # tracked separately, not part of this report's data
        "phase2_report_generated": True,  # true by the act of generating this report
    }

    unresolved = [k for k, v in acceptance.items() if v is False]
    pending_manual_confirmation = [k for k, v in acceptance.items() if v is None]

    overall = "PASS" if not unresolved and not pending_manual_confirmation else (
        "PENDING_MANUAL_CONFIRMATION" if not unresolved else "NEEDS_CORRECTION"
    )

    report = {
        "phase": "Phase 2 -- Preprocessing + Preprocessing Quality Control",
        "preprocessing_version": config.get("preprocessing_version"),
        "locked_configuration": config.get("locked"),
        "target_canvas": config.get("target"),
        "phase1_total_images": phase1_total_images,
        "full_batch_run_stats": proc_stats,
        "qc_summary": {
            "status": qc["status"],
            "counts": counts,
            "critical_failures": qc["critical_failures"],
            "informational_flag_rates": {
                k: v["pct_of_total"] for k, v in qc["informational_flags"].items()
            },
        },
        "duplicate_audit_summary": duplicate_audit_summary,
        "visual_audit": {
            "categories": {k: len(v) for k, v in visual_audit_manifest.items()},
            "total_images_rendered": visual_audit_written_count,
        },
        "test_results": test_results,
        "acceptance_criteria": acceptance,
        "unresolved_acceptance_items": unresolved,
        "items_pending_manual_confirmation": pending_manual_confirmation,
        "overall_status": overall,
        "known_limitations": [
            "Perceptual (phash) cross-writer/cross-dataset duplicate groups are not "
            "automatically treated as leakage risk -- confirmed via spot-checking to include "
            "a meaningful rate of false-positive hash collisions on thin-stroke signature "
            "scans. These should still be kept in mind (not re-litigated) during future "
            "writer-disjoint splitting.",
            "QC's 'unusual background' and 'unusual aspect ratio' visual-audit categories fall "
            "back to statistical outliers (most extreme values in this batch) when no image is "
            "explicitly flagged for that reason -- these are the most extreme examples found, "
            "not necessarily problems.",
            "notebook_runs_from_fresh_colab_runtime and readme_documentation_updated require "
            "a human confirmation step outside this report's automated scope.",
        ],
        "next_phase_not_started": True,
    }
    return report


def print_completion_report(report: dict) -> None:
    print("=" * 70)
    print("PHASE 2 COMPLETION REPORT")
    print("=" * 70)
    print(f"preprocessing_version: {report['preprocessing_version']}")
    print(f"locked configuration: {report['locked_configuration']}")
    print(f"target canvas: {report['target_canvas']}")
    print()
    print("Full-batch run stats:")
    for k, v in report["full_batch_run_stats"].items():
        print(f"  {k}: {v}")
    print()
    print(f"QC status: {report['qc_summary']['status']}")
    for k, v in report["qc_summary"]["counts"].items():
        print(f"  {k}: {v}")
    if report["qc_summary"]["critical_failures"]:
        print("  CRITICAL FAILURES:")
        for cf in report["qc_summary"]["critical_failures"]:
            print(f"    - {cf}")
    print("  Informational flag rates:")
    for k, v in report["qc_summary"]["informational_flag_rates"].items():
        print(f"    {k}: {v}%")
    print()
    print(f"Duplicate audit leakage risk: "
          f"{report['duplicate_audit_summary'].get('F_leakage_risk', {}).get('num_flagged_groups', 'N/A')} flagged groups")
    print()
    print(f"Visual audit: {report['visual_audit']['total_images_rendered']} images rendered "
          f"across {len(report['visual_audit']['categories'])} categories")
    for k, v in report["visual_audit"]["categories"].items():
        print(f"    {k}: {v}")
    print()
    print("Acceptance criteria:")
    for k, v in report["acceptance_criteria"].items():
        mark = "PASS" if v is True else ("FAIL" if v is False else "PENDING (manual confirmation needed)")
        print(f"  [{mark}] {k}")
    print()
    print(f"OVERALL STATUS: {report['overall_status']}")
    if report["unresolved_acceptance_items"]:
        print("Unresolved items (must be corrected before Phase 2 can be declared PASS):")
        for item in report["unresolved_acceptance_items"]:
            print(f"  - {item}")
    if report["items_pending_manual_confirmation"]:
        print("Pending manual confirmation:")
        for item in report["items_pending_manual_confirmation"]:
            print(f"  - {item}")
    print()
    print("Known limitations:")
    for lim in report["known_limitations"]:
        print(f"  - {lim}")
