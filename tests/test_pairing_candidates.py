import pandas as pd
import pytest

from pairing.candidates import (
    build_exact_duplicate_pairs,
    make_pair_id,
    generate_writer_pairs,
    generate_all_candidate_pairs,
)


def _manifest(rows):
    return pd.DataFrame(rows)


def _image_metadata(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# make_pair_id / build_exact_duplicate_pairs
# --------------------------------------------------------------------------

def test_make_pair_id_deterministic():
    assert make_pair_id("CEDAR_1", "genuine_genuine", "A", "B") == make_pair_id("CEDAR_1", "genuine_genuine", "A", "B")


def test_build_exact_duplicate_pairs_groups_correctly():
    dup_df = pd.DataFrame([
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": "A"},
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": "B"},
        {"hash_type": "sha256_exact", "hash_value": "h1", "image_id": "C"},
        {"hash_type": "phash_perceptual", "hash_value": "h2", "image_id": "D"},
        {"hash_type": "phash_perceptual", "hash_value": "h2", "image_id": "E"},
    ])
    pairs = build_exact_duplicate_pairs(dup_df)
    assert frozenset(("A", "B")) in pairs
    assert frozenset(("A", "C")) in pairs
    assert frozenset(("B", "C")) in pairs
    assert frozenset(("D", "E")) not in pairs  # perceptual, not exact -- must not be included


def test_build_exact_duplicate_pairs_handles_none_and_empty():
    assert build_exact_duplicate_pairs(None) == set()
    assert build_exact_duplicate_pairs(pd.DataFrame(columns=["hash_type", "hash_value", "image_id"])) == set()


# --------------------------------------------------------------------------
# generate_writer_pairs: canonical construction, dedup, exclusion
# --------------------------------------------------------------------------

def test_generate_writer_pairs_genuine_genuine_count():
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2", "G3"], [], set())
    gg = [r for r in rows if r["pair_type"] == "genuine_genuine"]
    assert len(gg) == 3  # C(3,2)


def test_generate_writer_pairs_genuine_forgery_count():
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2"], ["F1", "F2", "F3"], set())
    gf = [r for r in rows if r["pair_type"] == "genuine_forgery"]
    assert len(gf) == 6  # 2*3


def test_generate_writer_pairs_no_forgery_forgery_pairs():
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2"], ["F1", "F2"], set())
    assert all(r["pair_type"] in ("genuine_genuine", "genuine_forgery") for r in rows)


def test_generate_writer_pairs_labels_correct():
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2"], ["F1"], set())
    for r in rows:
        if r["pair_type"] == "genuine_genuine":
            assert r["pair_label"] == 1
        else:
            assert r["pair_label"] == 0


def test_generate_writer_pairs_genuine_forgery_image1_always_genuine():
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2"], ["F1"], set())
    gf = [r for r in rows if r["pair_type"] == "genuine_forgery"]
    assert all(r["image_1_id"].startswith("G") for r in gf)
    assert all(r["image_2_id"].startswith("F") for r in gf)


def test_generate_writer_pairs_excludes_exact_duplicate_self_pair():
    exact_dup_pairs = {frozenset(("G1", "G2"))}
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2", "G3"], [], exact_dup_pairs)
    gg_pairs = {frozenset((r["image_1_id"], r["image_2_id"])) for r in rows}
    assert frozenset(("G1", "G2")) not in gg_pairs
    assert frozenset(("G1", "G3")) in gg_pairs
    assert frozenset(("G2", "G3")) in gg_pairs


def test_generate_writer_pairs_no_reversed_duplicates():
    rows = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2", "G3"], ["F1", "F2"], set())
    seen = set()
    for r in rows:
        key = (r["pair_type"], frozenset((r["image_1_id"], r["image_2_id"])))
        assert key not in seen
        seen.add(key)


def test_generate_writer_pairs_deterministic_regardless_of_input_order():
    r1 = generate_writer_pairs("W1", "CEDAR", "train", ["G3", "G1", "G2"], ["F2", "F1"], set())
    r2 = generate_writer_pairs("W1", "CEDAR", "train", ["G1", "G2", "G3"], ["F1", "F2"], set())
    ids1 = sorted(r["pair_id"] for r in r1)
    ids2 = sorted(r["pair_id"] for r in r2)
    assert ids1 == ids2


def test_generate_writer_pairs_pair_id_contains_writer_and_type():
    rows = generate_writer_pairs("CEDAR_5", "CEDAR", "train", ["G1", "G2"], ["F1"], set())
    for r in rows:
        assert "CEDAR_5" in r["pair_id"]
        assert r["pair_type"] in r["pair_id"]


def test_generate_writer_pairs_split_and_writer_fields_consistent():
    rows = generate_writer_pairs("W1", "CEDAR", "val", ["G1", "G2"], ["F1"], set())
    assert all(r["split"] == "val" for r in rows)
    assert all(r["global_writer_id"] == "W1" for r in rows)
    assert all(r["dataset"] == "CEDAR" for r in rows)


# --------------------------------------------------------------------------
# generate_all_candidate_pairs: eligibility filtering, split isolation
# --------------------------------------------------------------------------

@pytest.fixture
def two_writer_setup():
    manifest = _manifest([
        {"global_writer_id": "CEDAR_1", "dataset": "CEDAR", "split": "train", "pair_generation_eligible": True},
        {"global_writer_id": "BHSIG_HINDI_123", "dataset": "BHSIG_HINDI", "split": "train", "pair_generation_eligible": False},
    ])
    images = _image_metadata([
        {"image_id": "C1_G1", "global_writer_id": "CEDAR_1", "signature_type": "genuine"},
        {"image_id": "C1_G2", "global_writer_id": "CEDAR_1", "signature_type": "genuine"},
        {"image_id": "C1_F1", "global_writer_id": "CEDAR_1", "signature_type": "forgery"},
        {"image_id": "H123_F1", "global_writer_id": "BHSIG_HINDI_123", "signature_type": "forgery"},
        {"image_id": "H123_F2", "global_writer_id": "BHSIG_HINDI_123", "signature_type": "forgery"},
    ])
    return manifest, images


def test_generate_all_candidate_pairs_skips_ineligible_writer(two_writer_setup):
    manifest, images = two_writer_setup
    pairs = generate_all_candidate_pairs(manifest, images, None)
    assert (pairs["global_writer_id"] == "BHSIG_HINDI_123").sum() == 0


def test_generate_all_candidate_pairs_produces_pairs_for_eligible_writer(two_writer_setup):
    manifest, images = two_writer_setup
    pairs = generate_all_candidate_pairs(manifest, images, None)
    # CEDAR_1 has 2 genuine (C1_G1, C1_G2) and 1 forgery (C1_F1):
    # genuine_genuine = C(2,2) = 1, genuine_forgery = 2*1 = 2 -> 3 total
    assert (pairs["global_writer_id"] == "CEDAR_1").sum() == 3


def test_generate_all_candidate_pairs_never_mixes_writers(two_writer_setup):
    manifest, images = two_writer_setup
    pairs = generate_all_candidate_pairs(manifest, images, None)
    # structural guarantee -- every pair's images must belong to its own writer
    img_to_writer = images.set_index("image_id")["global_writer_id"].to_dict()
    for row in pairs.itertuples(index=False):
        assert img_to_writer[row.image_1_id] == row.global_writer_id
        assert img_to_writer[row.image_2_id] == row.global_writer_id


def test_generate_all_candidate_pairs_empty_manifest_returns_empty_df():
    manifest = _manifest([])  # pandas gives this zero columns, not even the expected ones
    manifest = pd.DataFrame(columns=["global_writer_id", "dataset", "split", "pair_generation_eligible"])
    images = pd.DataFrame(columns=["image_id", "global_writer_id", "signature_type"])
    pairs = generate_all_candidate_pairs(manifest, images, None)
    assert len(pairs) == 0
    assert list(pairs.columns) == ["pair_id", "image_1_id", "image_2_id", "global_writer_id",
                                    "dataset", "split", "pair_label", "pair_type"]


def test_generate_all_candidate_pairs_deterministic_rerun(two_writer_setup):
    manifest, images = two_writer_setup
    p1 = generate_all_candidate_pairs(manifest, images, None)
    p2 = generate_all_candidate_pairs(manifest, images, None)
    assert p1["pair_id"].tolist() == p2["pair_id"].tolist()
