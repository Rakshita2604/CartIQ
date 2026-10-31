import random

import numpy as np
import pytest

from cartiq.embeddings.encoder import l2_normalize
from cartiq.retrieval.dense import DenseIndex

RNG = np.random.default_rng(0)
IDS = [f"P{i:03d}" for i in range(50)]
VECTORS = l2_normalize(RNG.normal(size=(50, 16)).astype(np.float32))
QUERY = l2_normalize(RNG.normal(size=(1, 16)).astype(np.float32))[0]


def test_exact_top_k_matches_brute_force() -> None:
    hits = DenseIndex(IDS, VECTORS).search(QUERY, 10)
    expected = sorted(range(50), key=lambda i: (-float(VECTORS[i] @ QUERY), IDS[i]))[:10]

    assert [h.product_id for h in hits] == [IDS[i] for i in expected]
    assert [h.rank for h in hits] == list(range(1, 11))
    assert hits[0].score == pytest.approx(float(VECTORS[expected[0]] @ QUERY))  # raw cosine kept


def test_store_eligibility_applies_before_the_cut() -> None:
    index = DenseIndex(IDS, VECTORS)
    everything = [h.product_id for h in index.search(QUERY, 50)]
    allowed = set(everything[25:])  # only the weaker half is in stock

    hits = index.search(QUERY, 5, allowed=allowed)

    assert [h.product_id for h in hits] == everything[25:30]


def test_empty_store_and_unknown_ids() -> None:
    index = DenseIndex(IDS, VECTORS)

    assert index.search(QUERY, 5, allowed=set()) == []
    assert index.search(QUERY, 5, allowed={"nope"}) == []


def test_ties_and_near_ties_are_ordered_by_product_id() -> None:
    same = l2_normalize(np.array([[1.0, 0.0]] * 3, dtype=np.float32))
    same[1] += np.float32(1e-9)  # below the sort rounding: still a tie
    index = DenseIndex(["c", "a", "b"], same)

    assert [h.product_id for h in index.search(np.array([1.0, 0.0], dtype=np.float32), 3)] == [
        "a",
        "b",
        "c",
    ]


def test_input_order_does_not_matter() -> None:
    order = list(range(50))
    random.Random(2).shuffle(order)

    a = DenseIndex(IDS, VECTORS).search(QUERY, 20)
    b = DenseIndex([IDS[i] for i in order], VECTORS[order]).search(QUERY, 20)

    assert a == b


def test_depth_must_be_positive() -> None:
    with pytest.raises(ValueError, match="depth"):
        DenseIndex(IDS, VECTORS).search(QUERY, 0)
