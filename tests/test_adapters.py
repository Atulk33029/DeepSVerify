from dv_datasets.cedar import CedarAdapter
from dv_datasets.bhsig_bengali import BHSigBengaliAdapter
from dv_datasets.bhsig_hindi import BHSigHindiAdapter
from dv_datasets.common import SignatureType, DatasetName


def test_cedar_adapter_discovers_and_parses(cedar_root):
    adapter = CedarAdapter(cedar_root, {})
    records = list(adapter.discover())

    parsed = [r for r in records if r.parse_ok]
    unparsed = [r for r in records if not r.parse_ok]

    # 2 writers * 2 samples * 2 (genuine+forgery) = 8 parsed, + 1 unparsed
    assert len(parsed) == 8
    assert len(unparsed) == 1
    assert all(r.dataset == DatasetName.CEDAR for r in records)

    genuine = [r for r in parsed if r.signature_type == SignatureType.GENUINE]
    forgery = [r for r in parsed if r.signature_type == SignatureType.FORGERY]
    assert len(genuine) == 4
    assert len(forgery) == 4
    assert {r.writer_id for r in parsed} == {"1", "2"}


def test_cedar_adapter_fallback_scan(tmp_path):
    # No full_org/full_forg subfolders -> falls back to recursive scan
    root = tmp_path / "cedar_flat"
    from PIL import Image
    (root).mkdir()
    Image.new("RGB", (10, 10)).save(root / "original_3_1.png")
    adapter = CedarAdapter(root, {})
    records = list(adapter.discover())
    assert len(records) == 1
    assert records[0].parse_ok
    assert records[0].writer_id == "3"


def test_bhsig_bengali_adapter(bhsig_bengali_root):
    adapter = BHSigBengaliAdapter(bhsig_bengali_root, {})
    records = list(adapter.discover())
    assert len(records) == 8  # 2 writers * 2 samples * 2 types
    assert all(r.language == "Bengali" for r in records)
    assert all(r.dataset == DatasetName.BHSIG_BENGALI for r in records)
    assert {r.writer_id for r in records} == {"001", "002"}


def test_bhsig_hindi_adapter(bhsig_hindi_root):
    adapter = BHSigHindiAdapter(bhsig_hindi_root, {})
    records = list(adapter.discover())
    assert len(records) == 8
    assert all(r.language == "Hindi" for r in records)
    assert all(r.dataset == DatasetName.BHSIG_HINDI for r in records)


def test_cedar_adapter_writer_folders_layout(writer_folders_root):
    # Reuse the CedarAdapter purely as a generic writer_folders-layout
    # exerciser -- the layout/classification logic is dataset-agnostic.
    adapter = CedarAdapter(writer_folders_root, {"layout": "writer_folders"})
    records = list(adapter.discover())

    assert len(records) == 9  # 2 writers * 2 samples * 2 types + 1 mystery file

    by_writer = {}
    for r in records:
        by_writer.setdefault(r.writer_id, []).append(r)

    assert set(by_writer.keys()) == {"1", "2", "3"}

    writer1 = by_writer["1"]
    genuine = [r for r in writer1 if r.signature_type == SignatureType.GENUINE]
    forgery = [r for r in writer1 if r.signature_type == SignatureType.FORGERY]
    assert len(genuine) == 2
    assert len(forgery) == 2
    assert all(r.parse_ok for r in writer1)

    mystery = by_writer["3"][0]
    assert mystery.parse_ok is False
    assert mystery.signature_type == "unknown"


def test_bhsig_bengali_writer_folders_layout(writer_folders_root):
    adapter = BHSigBengaliAdapter(writer_folders_root, {"layout": "writer_folders"})
    records = list(adapter.discover())
    assert len(records) == 9
    assert all(r.language == "Bengali" for r in records)


def test_inspect_produces_report(cedar_root):
    adapter = CedarAdapter(cedar_root, {})
    result = adapter.inspect()
    d = result.to_dict()
    assert d["total_files"] == 9
    assert d["valid_images"] == 9  # all are structurally valid PNGs, even the unparsed one
    assert d["unparsed_files"] == 1
    assert d["num_writers"] == 2
