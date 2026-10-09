"""
pairing/report.py

Assembles the final Phase 3B report covering every item the spec's
"FINAL REPORT REQUIRED" section asks for. Reads only already-computed
results -- makes no decisions itself.
"""

from __future__ import annotations

import pandas as pd


def derive_known_data_characteristics(qc_result: dict, manifest_df: pd.DataFrame,
                                      verified_notes: dict[str, str] | None = None) -> list[dict]:
    """Builds the 'known data characteristics' list mechanically from the
    QC distribution report's existing statistical outlier detection
    (writers_with_unusually_high_count / _low_count -- mean +/- 2*std,
    see pairing/qc.py), joined against the real manifest for that
    writer's actual n_genuine/n_forgery. No writer_id is special-cased in
    code; outliers are whichever writers the measured distribution
    flags.

    verified_notes: optional {global_writer_id: note} to attach a
    human-verified explanation (e.g. "confirmed via raw-folder
    inspection: ...") to a specific flagged writer, once investigated.
    Writers without a note still appear, marked as unverified.
    """
    dr = qc_result["distribution_report"]
    manifest_lookup = manifest_df.set_index("global_writer_id")
    notes = verified_notes or {}

    entries = []
    for direction, outliers in (("unusually_high", dr.get("writers_with_unusually_high_count", {})),
                               ("unusually_low", dr.get("writers_with_unusually_low_count", {}))):
        for writer_id, pair_count in outliers.items():
            row = manifest_lookup.loc[writer_id] if writer_id in manifest_lookup.index else None
            entries.append({
                "global_writer_id": writer_id,
                "dataset": row["dataset"] if row is not None else None,
                "split": row["split"] if row is not None else None,
                "n_genuine": int(row["n_genuine"]) if row is not None else None,
                "n_forgery": int(row["n_forgery"]) if row is not None else None,
                "total_pairs_generated": int(pair_count),
                "flagged_as": direction,
                "status": "verified_source_data_characteristic" if writer_id in notes else "unverified_outlier",
                "note": notes.get(writer_id, "Statistical outlier flagged by QC; not yet manually verified against raw source files."),
            })
    return entries


def build_generation_report(*, config: dict, measurement_agg: dict, sampling_summary: dict,
                            qc_result: dict, pairs_df: pd.DataFrame, test_results: dict | None = None,
                            known_data_characteristics: list[dict] | None = None) -> dict:
    hindi_123 = pairs_df[pairs_df["global_writer_id"] == "BHSIG_HINDI_123"]

    acceptance = {
        "pairing_methodology_documented": True,
        "pairs_actually_generated": len(pairs_df) > 0,
        "pair_level_qc_passed": qc_result["status"] == "PASS",
        "tests_pass": (test_results or {}).get("all_passed"),
        "reproducibility_demonstrated": None,  # set by caller after an actual rerun comparison
        "outputs_saved": None,  # set by caller once files are written
        "no_split_leakage": qc_result["critical_checks"].get("no_pair_crosses_split", False),
        "no_ineligible_writer_pairs": qc_result["critical_checks"].get("no_ineligible_writer_produces_pairs", False),
        "distribution_statistics_reported": True,
    }
    unresolved = [k for k, v in acceptance.items() if v is False]
    pending = [k for k, v in acceptance.items() if v is None]
    overall = "NEEDS_CORRECTION" if unresolved else ("PENDING_CONFIRMATION" if pending else "PASS")

    return {
        "phase": "Phase 3B -- Pair Generation",
        "pair_generation_version": config.get("pair_generation_version"),
        "seed": config.get("seed"),
        "negative_to_positive_ratio_configured": config.get("negative_to_positive_ratio"),
        "pairing_strategy": {
            "positive_pairs": "genuine-genuine, same writer, ALL kept (no subsampling)",
            "negative_pairs": "genuine-forgery, same writer, per-writer capped at ratio x positive_count, deterministic seeded subsample when exceeding cap",
            "forgery_forgery_pairs": "not generated (per explicit instruction; no demonstrated methodological need)",
            "exact_duplicate_handling": (
                "Duplicate images kept as individual usable samples; only the specific "
                "genuine_genuine pair where both images are each other's exact (sha256) "
                "duplicate is excluded (zero-distance, uninformative). genuine_forgery pairs "
                "never need this filter -- Phase 2 found zero exact-duplicate groups mix "
                "genuine and forgery labels."
            ),
            "canonical_ordering": "image_1 = genuine (reference), image_2 = genuine or forgery (query); prevents reversed duplicates structurally",
        },
        "measurement": measurement_agg,
        "sampling": sampling_summary,
        "qc": {
            "status": qc_result["status"],
            "critical_checks": qc_result["critical_checks"],
            "critical_failures": qc_result["critical_failures"],
            "perceptual_cross_split_pair_overlap": qc_result["perceptual_cross_split_pair_overlap"],
            "distribution_report": qc_result["distribution_report"],
        },
        "bhsig_hindi_123": {
            "pairs_generated": int(len(hindi_123)),
            "expected": 0,
            "matches_expectation": len(hindi_123) == 0,
        },
        "known_data_characteristics": known_data_characteristics or [],
        "test_results": test_results,
        "acceptance_criteria": acceptance,
        "unresolved_acceptance_items": unresolved,
        "items_pending_confirmation": pending,
        "overall_status": overall,
        "next_phase_not_started": True,
    }


def print_generation_report(report: dict) -> None:
    print("=" * 70)
    print("PHASE 3B PAIR GENERATION REPORT")
    print("=" * 70)
    print(f"seed: {report['seed']}  negative_to_positive_ratio: {report['negative_to_positive_ratio_configured']}")
    print("\nStrategy:")
    for k, v in report["pairing_strategy"].items():
        print(f"  {k}: {v}")

    m = report["measurement"]
    print(f"\nTheoretical (pre-sampling) totals: "
          f"positive={m['total_possible_positive_pairs']:,}  negative={m['total_possible_negative_pairs']:,}")

    s = report["sampling"]
    print(f"Final totals after sampling: "
          f"positive={s['final_counts']['positive']:,}  negative={s['final_counts']['negative']:,}  "
          f"ratio={s['final_negative_to_positive_ratio']}")
    print(f"Negatives discarded: {s['negatives_discarded']:,} ({s['pct_negatives_discarded']}%)")

    print(f"\nQC status: {report['qc']['status']}")
    if report["qc"]["critical_failures"]:
        print(f"  Critical failures: {report['qc']['critical_failures']}")

    h = report["bhsig_hindi_123"]
    print(f"\nBHSIG_HINDI_123: {h['pairs_generated']} pairs generated "
          f"(expected 0) -- {'OK' if h['matches_expectation'] else 'MISMATCH'}")

    if report["known_data_characteristics"]:
        print("\nKnown data characteristics (writer-level outliers, mechanically flagged by QC):")
        for c in report["known_data_characteristics"]:
            print(f"  - {c['global_writer_id']} ({c['dataset']}, split={c['split']}): "
                  f"n_genuine={c['n_genuine']}, n_forgery={c['n_forgery']}, "
                  f"{c['total_pairs_generated']} pairs generated ({c['flagged_as']})")
            print(f"      status: {c['status']}")
            print(f"      note: {c['note']}")

    print(f"\nOVERALL STATUS: {report['overall_status']}")
    if report["unresolved_acceptance_items"]:
        print(f"Unresolved: {report['unresolved_acceptance_items']}")
    if report["items_pending_confirmation"]:
        print(f"Pending confirmation: {report['items_pending_confirmation']}")
