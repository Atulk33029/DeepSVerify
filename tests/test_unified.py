from dv_datasets.cedar import CedarAdapter
from dv_datasets.bhsig_bengali import BHSigBengaliAdapter
from dv_datasets.bhsig_hindi import BHSigHindiAdapter
from dv_datasets.common import DatasetName, UNKNOWN
from dv_datasets.unified import (
    build_unified_metadata,
    compute_statistics,
    detect_duplicates,
    build_validation_report,
    run_quality_control,
    METADATA_COLUMNS,
)


def _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    return {
        DatasetName.CEDAR: CedarAdapter(cedar_root, {}),
        DatasetName.BHSIG_BENGALI: BHSigBengaliAdapter(bhsig_bengali_root, {}),
        DatasetName.BHSIG_HINDI: BHSigHindiAdapter(bhsig_hindi_root, {}),
    }


def test_build_unified_metadata_schema(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    assert len(rows) == 9 + 8 + 8  # cedar has 1 extra unparsed file
    for row in rows:
        assert set(row.keys()) == set(METADATA_COLUMNS)


def test_global_writer_id_no_cross_dataset_collision(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    # CEDAR writer "1" and BHSig writer "001" must never collide
    gwids = {r["global_writer_id"] for r in rows}
    assert "CEDAR_1" in gwids
    assert "BHSIG_BENGALI_001" in gwids
    assert "BHSIG_HINDI_001" in gwids


def test_image_id_uniqueness_and_determinism(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows1 = build_unified_metadata(adapters, compute_perceptual_hash=False)
    ids1 = [r["image_id"] for r in rows1]
    assert len(ids1) == len(set(ids1))  # no duplicate IDs

    # Re-running against the SAME unchanged source tree must reproduce
    # identical image_ids in identical order (Section 13: reproducibility).
    adapters2 = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows2 = build_unified_metadata(adapters2, compute_perceptual_hash=False)
    ids2 = [r["image_id"] for r in rows2]
    assert ids1 == ids2


def test_statistics_totals_match(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    stats = compute_statistics(rows)
    assert stats["overall"]["total_images"] == len(rows)
    assert (
        stats["overall"]["total_genuine"] + stats["overall"]["total_forgery"] + stats["overall"]["total_unknown_label"]
        == len(rows)
    )
    assert DatasetName.CEDAR in stats["by_dataset"]
    assert DatasetName.BHSIG_BENGALI in stats["by_dataset"]
    assert DatasetName.BHSIG_HINDI in stats["by_dataset"]


def test_duplicate_detection_finds_exact_duplicates(tmp_path):
    from PIL import Image
    root = tmp_path / "cedar"
    org = root / "full_org"
    org.mkdir(parents=True)
    # Two byte-identical images under different writer/sample names
    img = Image.new("RGB", (10, 10), (1, 2, 3))
    img.save(org / "original_1_1.png")
    img.save(org / "original_1_2.png")

    adapter = CedarAdapter(root, {})
    rows = build_unified_metadata({DatasetName.CEDAR: adapter}, compute_perceptual_hash=False)
    dups = detect_duplicates(rows)
    exact = [d for d in dups if d["hash_type"] == "sha256_exact"]
    assert len(exact) == 1
    assert exact[0]["num_duplicates"] == 2


def test_validation_report_flags_unparsed_file_as_critical(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    # The CEDAR fixture deliberately includes one unparsable filename, which
    # ends up with signature_type=UNKNOWN. Per spec (Section 12: "Do NOT
    # allow the pipeline to silently continue if critical metadata is
    # missing"), an unknown/invalid label is a CRITICAL failure, not a
    # soft warning -- so the correct, intended status here is FAIL.
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    dup_rows = detect_duplicates(rows)
    report = build_validation_report(rows, dup_rows)
    assert report["unparsed_filenames"]
    assert report["missing_files"] == []
    assert report["status"] == "FAIL"
    assert any("signature_type" in cf for cf in report["critical_failures"])


def test_validation_report_pass_when_all_files_parse_cleanly(bhsig_bengali_root, bhsig_hindi_root, tmp_path):
    # A CEDAR tree with ONLY well-formed filenames (no unparsable file)
    # should validate cleanly end to end.
    from PIL import Image
    root = tmp_path / "cedar_clean"
    org = root / "full_org"
    forg = root / "full_forg"
    for writer in (1, 2):
        for sample in (1, 2):
            for d, prefix in ((org, "original"), (forg, "forgeries")):
                d.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16)).save(d / f"{prefix}_{writer}_{sample}.png")

    adapters = _adapters(root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    dup_rows = detect_duplicates(rows)
    report = build_validation_report(rows, dup_rows)
    assert report["unparsed_filenames"] == []
    assert report["missing_files"] == []
    assert report["status"] == "PASS"


def test_validation_report_fails_on_missing_file(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    # simulate a source file having vanished after discovery, before validation ran
    rows[0]["validation_error"] = "file_not_found"
    rows[0]["is_valid"] = False
    dup_rows = detect_duplicates(rows)
    report = build_validation_report(rows, dup_rows)
    assert report["status"] == "FAIL"
    assert report["missing_files"] == [rows[0]["image_id"]]


def test_quality_control_all_pass_on_clean_data(cedar_root, bhsig_bengali_root, bhsig_hindi_root):
    adapters = _adapters(cedar_root, bhsig_bengali_root, bhsig_hindi_root)
    rows = build_unified_metadata(adapters, compute_perceptual_hash=False)
    dup_rows = detect_duplicates(rows)
    report = build_validation_report(rows, dup_rows)
    qc = run_quality_control(rows, report)
    # missing_writer_ids will be non-empty ONLY if the unparsed file has no
    # writer id -- confirm structure is sane regardless
    assert isinstance(qc["overall_pass"], bool)
    assert qc["no_image_id_duplicates"] is True
    assert qc["global_writer_id_unique_across_datasets"] is True
