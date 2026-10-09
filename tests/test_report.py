from preprocessing.report import build_completion_report


def _clean_qc():
    return {
        "status": "PASS",
        "critical_failures": [],
        "counts": {
            "phase1_valid_images": 16680, "preprocessing_metadata_rows": 16680,
            "status_ok": 16500, "status_flagged": 175, "status_failed": 5,
        },
        "critical_checks": {"output_count_matches_valid_input_count": True},
        "informational_flags": {
            "tiny_foreground": {"count": 50, "pct_of_total": 0.3, "sample_image_ids": []},
        },
    }


def _clean_dup_summary():
    return {"F_leakage_risk": {"num_flagged_groups": 0, "num_flagged_images": 0}}


def _config():
    return {
        "preprocessing_version": "phase2-full-v1",
        "locked": {"intensity_variant": "flatten_bg", "margin_fraction": 0.10},
        "target": {"width": 384, "height": 128, "pad_value": 255},
    }


def test_report_overall_pass_when_everything_clean():
    report = build_completion_report(
        phase1_total_images=16680,
        proc_stats={"total_valid_input_images": 16680, "newly_processed": 16680,
                    "resumed_from_previous_run": 0, "failed": 5},
        qc=_clean_qc(),
        duplicate_audit_summary=_clean_dup_summary(),
        visual_audit_manifest={"typical__CEDAR__genuine": ["a", "b"]},
        visual_audit_written_count=30,
        config=_config(),
        test_results={"all_passed": True, "n_tests": 100},
    )
    # notebook_runs_from_fresh_colab_runtime and readme_documentation_updated
    # are always None (require human confirmation) -- so overall status
    # should be PENDING, not an outright PASS, until a human confirms those.
    assert report["overall_status"] == "PENDING_MANUAL_CONFIRMATION"
    assert report["unresolved_acceptance_items"] == []
    assert "notebook_runs_from_fresh_colab_runtime" in report["items_pending_manual_confirmation"]


def test_report_needs_correction_on_qc_failure():
    qc = _clean_qc()
    qc["status"] = "FAIL"
    qc["critical_failures"] = ["no_missing_outputs"]
    report = build_completion_report(
        phase1_total_images=16680,
        proc_stats={"total_valid_input_images": 16680, "newly_processed": 16680,
                    "resumed_from_previous_run": 0, "failed": 5},
        qc=qc,
        duplicate_audit_summary=_clean_dup_summary(),
        visual_audit_manifest={"a": ["x"]},
        visual_audit_written_count=10,
        config=_config(),
    )
    assert report["overall_status"] == "NEEDS_CORRECTION"
    assert "qc_passes" in report["unresolved_acceptance_items"]


def test_report_needs_correction_on_leakage_risk():
    dup_summary = {"F_leakage_risk": {"num_flagged_groups": 3, "num_flagged_images": 7}}
    report = build_completion_report(
        phase1_total_images=16680,
        proc_stats={"total_valid_input_images": 16680, "newly_processed": 16680,
                    "resumed_from_previous_run": 0, "failed": 0},
        qc=_clean_qc(),
        duplicate_audit_summary=dup_summary,
        visual_audit_manifest={"a": ["x"]},
        visual_audit_written_count=10,
        config=_config(),
    )
    assert report["overall_status"] == "NEEDS_CORRECTION"
    assert "no_unresolved_critical_leakage" in report["unresolved_acceptance_items"]


def test_report_flags_missing_visual_audit():
    report = build_completion_report(
        phase1_total_images=16680,
        proc_stats={"total_valid_input_images": 16680, "newly_processed": 16680,
                    "resumed_from_previous_run": 0, "failed": 0},
        qc=_clean_qc(),
        duplicate_audit_summary=_clean_dup_summary(),
        visual_audit_manifest={},
        visual_audit_written_count=0,
        config=_config(),
    )
    assert "visual_audit_generated" in report["unresolved_acceptance_items"]
