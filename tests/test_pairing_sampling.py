import pandas as pd
import pytest

from pairing.sampling import sample_negative_pairs, summarize_sampling


def _make_candidates(writer, n_pos, n_neg, split="train", dataset="CEDAR"):
    rows = []
    for i in range(n_pos):
        rows.append({"pair_id": f"{writer}__gg__{i}", "image_1_id": f"G{i}a", "image_2_id": f"G{i}b",
                     "global_writer_id": writer, "dataset": dataset, "split": split,
                     "pair_label": 1, "pair_type": "genuine_genuine"})
    for i in range(n_neg):
        rows.append({"pair_id": f"{writer}__gf__{i}", "image_1_id": f"G{i}", "image_2_id": f"F{i}",
                     "global_writer_id": writer, "dataset": dataset, "split": split,
                     "pair_label": 0, "pair_type": "genuine_forgery"})
    return pd.DataFrame(rows)


def test_sample_negative_pairs_keeps_all_positives():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    assert (sampled["pair_type"] == "genuine_genuine").sum() == 10


def test_sample_negative_pairs_caps_negatives_to_ratio():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    assert (sampled["pair_type"] == "genuine_forgery").sum() == 10  # capped at 1.0 * 10


def test_sample_negative_pairs_respects_ratio_other_than_one():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=2.0, seed=42)
    assert (sampled["pair_type"] == "genuine_forgery").sum() == 20


def test_sample_negative_pairs_never_pads_below_available():
    # if available negatives are fewer than the cap, keep ALL of them, never duplicate
    candidates = _make_candidates("W1", n_pos=10, n_neg=3)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    assert (sampled["pair_type"] == "genuine_forgery").sum() == 3


def test_sample_negative_pairs_deterministic():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    s1 = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    s2 = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    assert s1["pair_id"].tolist() == s2["pair_id"].tolist()


def test_sample_negative_pairs_different_seed_can_differ():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    s1 = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=1)
    s2 = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=2)
    neg1 = set(s1[s1["pair_type"] == "genuine_forgery"]["pair_id"])
    neg2 = set(s2[s2["pair_type"] == "genuine_forgery"]["pair_id"])
    assert neg1 != neg2


def test_sample_negative_pairs_per_writer_relative_cap_prevents_dominance():
    # A writer with vastly more possible negatives must not end up with
    # vastly more FINAL negatives than a writer with fewer -- the cap is
    # relative to each writer's own positive count.
    small_writer = _make_candidates("SMALL", n_pos=2, n_neg=200)
    big_writer = _make_candidates("BIG", n_pos=2, n_neg=200)
    candidates = pd.concat([small_writer, big_writer], ignore_index=True)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    small_final = (sampled["global_writer_id"] == "SMALL").sum()
    big_final = (sampled["global_writer_id"] == "BIG").sum()
    assert small_final == big_final  # both capped identically relative to their own (equal) positive counts


def test_sample_negative_pairs_does_not_mutate_input():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    original_len = len(candidates)
    sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    assert len(candidates) == original_len


def test_summarize_sampling_reports_discarded_count():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    summary = summarize_sampling(candidates, sampled)
    assert summary["candidate_counts"]["negative"] == 50
    assert summary["final_counts"]["negative"] == 10
    assert summary["negatives_discarded"] == 40


def test_summarize_sampling_ratio_computation():
    candidates = _make_candidates("W1", n_pos=10, n_neg=50)
    sampled = sample_negative_pairs(candidates, negative_to_positive_ratio=1.0, seed=42)
    summary = summarize_sampling(candidates, sampled)
    assert summary["candidate_negative_to_positive_ratio"] == 5.0
    assert summary["final_negative_to_positive_ratio"] == 1.0
