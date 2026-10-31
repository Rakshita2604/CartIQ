from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from cartiq.domain import (
    Constraint,
    Grade,
    JudgedQuery,
    MatchTier,
    QueryAttribute,
    QuerySegment,
    ScoredProduct,
)
from cartiq.evaluation.dense_eval import (
    CombinationResult,
    LatencyResult,
    Similarity,
    _complementarity,
    _similarity,
    load_plan,
    select,
)
from cartiq.retrieval.dense import DenseHit

PLAN, PLAN_SHA = load_plan(Path("config/dense-bakeoff-v1.toml"))

QUERY = JudgedQuery(
    "q1",
    "doodh",
    QuerySegment.HINGLISH,
    {"a": Grade.EXACT, "b": Grade.RELEVANT, "c": Grade.EXACT, "z": Grade.IRRELEVANT},
    "test",
)
EMPTY = JudgedQuery(
    "q2",
    "xyz",
    QuerySegment.HEAD,
    {"z": Grade.IRRELEVANT},
    "test",
    attributes=frozenset({QueryAttribute.NO_EXACT_MATCH}),
)


def classify(query: str, product_ids: Sequence[str]) -> Mapping[str, ScoredProduct]:
    """Stand-in for S6: "c" breaks the brand, everything else satisfies the query."""
    return {
        pid: ScoredProduct(pid, 0.0, MatchTier.RELAXED, frozenset({Constraint.BRAND}))
        if pid == "c"
        else ScoredProduct(pid, 0.0, MatchTier.EXACT)
        for pid in product_ids
    }


def hits(*ids: str, top: float = 0.9) -> list[DenseHit]:
    return [DenseHit(pid, top - 0.1 * i, i + 1) for i, pid in enumerate(ids)]


def test_frozen_plan_loads() -> None:
    assert PLAN.version == "dense-bakeoff-v1"
    assert PLAN.models == ["e5-small", "minilm-l12", "bge-small"]
    assert PLAN.judgments == "eval/judgments/synthetic-v1.toml"
    assert len(PLAN_SHA) == 64


def test_plan_refuses_held_out_judgments(tmp_path: Path) -> None:
    text = Path("config/dense-bakeoff-v1.toml").read_text()
    plan = tmp_path / "plan.toml"
    plan.write_text(text.replace("synthetic-v1.toml", "synthetic-v1-heldout.toml"))

    with pytest.raises(ValueError, match="development judgments only"):
        load_plan(plan)


def test_complementarity_separates_grades_and_computes_union_recall() -> None:
    dense = {"doodh": hits("z", "b", "c"), "xyz": hits("z")}
    reference = {"doodh": ["a"], "xyz": []}

    result = _complementarity("S0", PLAN, [QUERY, EMPTY], dense, reference, classify)

    assert [(c.product, c.dense_rank) for c in result.useful_missed_grade3] == [("c", 3)]
    assert [(c.product, c.dense_rank) for c in result.useful_missed_grade2] == [("b", 2)]
    assert result.queries_with_additions == 1
    assert result.queries_without_additions == ["xyz"]
    # Every dense-only candidate is kept, graded or not, with S6's tier.
    assert [(d.query, d.product, d.grade, d.s6_tier, d.s6_relaxed) for d in result.dense_only] == [
        ("doodh", "z", 0, "exact", []),
        ("doodh", "b", 2, "exact", []),
        ("doodh", "c", 3, "relaxed", ["brand"]),
        ("xyz", "z", 0, "exact", []),
    ]
    assert result.useful_s6_exact == 1  # "b"
    at10 = result.union_recall[10]
    # Only "doodh" has targets: reference finds 1 of 2 grade-3 and 1 of 3 grade-2+.
    assert (at10.reference_grade3, at10.union_grade3) == (0.5, 1.0)
    assert at10.reference_grade2plus == pytest.approx(1 / 3)
    assert at10.union_grade2plus == 1.0


def test_similarity_splits_by_relevance_and_segment() -> None:
    dense = {"doodh": hits("a", "b", "c", "z", "y"), "xyz": hits("z", top=0.4)}

    result: Similarity = _similarity(PLAN, [QUERY, EMPTY], dense)

    assert result.by_relevance["with_relevant"][1].maximum == pytest.approx(0.9)
    assert result.by_relevance["with_relevant"][5].maximum == pytest.approx(0.5)
    assert result.by_relevance["without_relevant"][1].n == 1
    assert 5 not in result.by_relevance["without_relevant"]  # fewer than 5 hits
    assert set(result.by_segment) == {"hinglish", "head"}


def combo(model: str, recall: float, ndcg: float) -> CombinationResult:
    return CombinationResult(
        model=model,
        query_template="raw-v1",
        product_template="base-v1",
        pipeline=f"dense-{model}-{ndcg}",
        artifact_id="x",
        encoder="e",
        catalog_sha256="c",
        vectors_sha256="v",
        overall={"recall@20": recall, "ndcg@10": ndcg},
        segments={},
        complementarity=[],
        misleading=[],
        similarity=Similarity(by_relevance={}, by_segment={}),
    )


def latency(model: str, eligible: bool) -> LatencyResult:
    return LatencyResult(model=model, queries=1, p50_ms=1, p95_ms=1, eligible=eligible)


def test_selection_rule_order() -> None:
    combos = [
        combo("e5-small", 0.8, 0.5),
        combo("bge-small", 0.8, 0.5),  # tie: the smaller model wins
        combo("minilm-l12", 0.9, 0.1),  # best recall but too slow
        combo("e5-small", 0.8, 0.6),
    ]
    lat = [latency("e5-small", True), latency("bge-small", True), latency("minilm-l12", False)]

    assert select(PLAN, combos, lat) == [
        "dense-e5-small-0.6",  # same recall, better NDCG
        "dense-bge-small-0.5",  # recall and NDCG tie: smaller model first
        "dense-e5-small-0.5",
    ]
