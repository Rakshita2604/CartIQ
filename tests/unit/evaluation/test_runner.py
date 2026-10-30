"""Runner behaviour on a two-store fixture with hand-worked expectations.

Stock: S1 has P1, P2, P3; S2 has only P2.
Queries:
  milk  (head, exhaustive):     P1=3, P2=2
  doodh (hinglish, partial):    P1=3, P4=0 explicit; P9 unjudged
  dahi  (hinglish, exhaustive): P3=3 (stocked only at S1)
  iphone(head, exhaustive, no_exact_match): nothing relevant exists
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from cartiq.domain import Coverage, Grade, JudgedQuery, QueryAttribute, QuerySegment
from cartiq.evaluation import EvalReport, JudgmentSet, StagedResults, evaluate, format_report
from cartiq.evaluation.runner import QueryResult

E, R, Z = Grade.EXACT, Grade.RELEVANT, Grade.IRRELEVANT  # Z: zero
IN_STOCK = {"S1": frozenset({"P1", "P2", "P3"}), "S2": frozenset({"P2"})}


def _query(
    text: str,
    segment: QuerySegment,
    grades: dict[str, Grade],
    coverage: Coverage = Coverage.EXHAUSTIVE,
    attributes: frozenset[QueryAttribute] = frozenset(),
) -> JudgedQuery:
    return JudgedQuery(text, text, segment, grades, "test", coverage, attributes)


JUDGMENTS = JudgmentSet(
    catalog="test-catalog",
    catalog_sha256="0" * 64,
    labeler="test",
    queries=(
        _query("milk", QuerySegment.HEAD, {"P1": E, "P2": R}),
        _query("doodh", QuerySegment.HINGLISH, {"P1": E, "P4": Z}, Coverage.PARTIAL),
        _query("dahi", QuerySegment.HINGLISH, {"P3": E}),
        _query(
            "iphone", QuerySegment.HEAD, {}, attributes=frozenset({QueryAttribute.NO_EXACT_MATCH})
        ),
    ),
    sha256="1" * 64,
)


@dataclass
class FakePipeline:
    """Canned results: retrieval per query, (available, final) per query and store."""

    retrieved: dict[str, list[str]]
    staged: dict[tuple[str, str], tuple[list[str], list[str]]] = field(default_factory=dict)
    name: str = "fake"

    def retrieve(self, query: str, depth: int) -> Sequence[str]:
        return self.retrieved.get(query, [])[:depth]

    def search(self, query: str, store_id: str, limit: int) -> StagedResults:
        available, final = self.staged.get((query, store_id), ([], []))
        return StagedResults(available=available, final=final[:limit])


def _both(results: list[str]) -> tuple[list[str], list[str]]:
    return results, results


PIPELINE = FakePipeline(
    retrieved={"milk": ["P1", "P2"], "doodh": ["P9"]},
    staged={
        ("milk", "S1"): _both(["P2", "P1"]),
        ("milk", "S2"): _both(["P2"]),
        ("doodh", "S1"): _both(["P4", "P9"]),
    },
)


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return evaluate(PIPELINE, JUDGMENTS, IN_STOCK, k=10, recall_at=(1, 10), snapshot_id="snap-t")


def _query_result(report: EvalReport, query_id: str) -> QueryResult:
    return next(q for q in report.queries if q.query_id == query_id)


def test_ideal_uses_only_products_in_stock_at_the_store(report: EvalReport) -> None:
    milk = _query_result(report, "milk")
    s1, s2 = milk.stores
    # S1: DCG = 3/log2(2) + 7/log2(3); IDCG = 7/log2(2) + 3/log2(3).
    assert s1.ndcg == pytest.approx((3 + 7 / math.log2(3)) / (7 + 3 / math.log2(3)))
    # S2 stocks only P2, so showing P2 alone is the best possible list.
    assert s2.ndcg == pytest.approx(1.0)
    assert (s1.reciprocal_rank, s2.reciprocal_rank) == (1.0, 1.0)
    assert s1.ndcg is not None
    assert milk.ndcg == pytest.approx((s1.ndcg + 1.0) / 2)


def test_whole_catalog_recall(report: EvalReport) -> None:
    assert _query_result(report, "milk").recall == {1: 0.5, 10: 1.0}
    assert _query_result(report, "doodh").recall == {1: 0.0, 10: 0.0}
    # Empty retrieval with a relevant product scores 0; nothing relevant is undefined.
    assert _query_result(report, "dahi").recall == {1: 0.0, 10: 0.0}
    assert _query_result(report, "iphone").recall == {1: None, 10: None}


def test_empty_results_score_zero_where_defined_and_none_elsewhere(report: EvalReport) -> None:
    dahi = _query_result(report, "dahi")
    s1, s2 = dahi.stores
    assert (s1.ndcg, s1.reciprocal_rank, s1.judged) == (0.0, 0.0, None)
    assert (s2.ndcg, s2.reciprocal_rank) == (None, None)  # P3 not stocked at S2
    assert dahi.ndcg == 0.0


def test_partial_judgments_keep_unjudged_separate(report: EvalReport) -> None:
    doodh = _query_result(report, "doodh")
    s1 = doodh.stores[0]
    assert [r.grade for r in s1.results] == [Grade.IRRELEVANT, None]
    assert s1.judged == pytest.approx(0.5)
    assert s1.ndcg == 0.0
    assert report.to_label == {"doodh": ["P9"]}


def test_zero_result_rates(report: EvalReport) -> None:
    overall = report.overall
    assert overall.catalog_zero_rate == pytest.approx(2 / 4)  # dahi, iphone
    # Per query over stores: milk 0, doodh 1/2, dahi 1, iphone 1.
    assert overall.available_zero_rate == pytest.approx((0 + 0.5 + 1 + 1) / 4)
    assert overall.final_zero_rate == pytest.approx(overall.available_zero_rate)
    assert report.warnings == []


def test_summary_excludes_and_counts_undefined(report: EvalReport) -> None:
    overall = report.overall
    milk_ndcg = _query_result(report, "milk").ndcg
    assert milk_ndcg is not None
    assert overall.queries == 4
    assert overall.partial_queries == 1
    assert overall.ndcg_undefined == 1  # iphone
    assert overall.ndcg == pytest.approx((milk_ndcg + 0 + 0) / 3)
    assert overall.recall_undefined == 1
    assert overall.recall[10] == pytest.approx((1 + 0 + 0) / 3)


def test_groups_by_segment_and_attribute(report: EvalReport) -> None:
    assert list(report.segments) == [QuerySegment.HEAD, QuerySegment.HINGLISH]
    assert report.segments[QuerySegment.HINGLISH].partial_queries == 1
    assert report.attributes[QueryAttribute.TRANSLITERATION].queries == 2
    assert report.attributes[QueryAttribute.NO_EXACT_MATCH].queries == 1


def test_report_is_deterministic_and_marks_estimates(report: EvalReport) -> None:
    again = evaluate(PIPELINE, JUDGMENTS, IN_STOCK, k=10, recall_at=(1, 10), snapshot_id="snap-t")
    table = format_report(report)

    assert again.model_dump_json() == report.model_dump_json()
    overall_row = next(line for line in table.splitlines() if line.startswith("overall"))
    assert "~" in overall_row
    head_row = next(line for line in table.splitlines() if line.startswith("seg:head"))
    assert "~" not in head_row


def test_hit_threshold_is_configurable() -> None:
    strict = evaluate(PIPELINE, JUDGMENTS, IN_STOCK, hit_min_grade=Grade.EXACT)
    milk = _query_result(strict, "milk")

    assert milk.stores[0].reciprocal_rank == 0.5  # P2 (grade 2) no longer a hit
    assert milk.stores[1].reciprocal_rank is None  # no grade 3 product stocked at S2


def test_warns_when_zero_result_rates_are_out_of_order() -> None:
    lossy = FakePipeline(
        retrieved={},  # retrieval finds nothing ...
        staged={(q.text, s): (["P2"], []) for q in JUDGMENTS.queries for s in IN_STOCK},
    )  # ... yet every store has available matches, and fallbacks drop them all.

    warnings = evaluate(lossy, JUDGMENTS, IN_STOCK).warnings

    assert warnings == [
        "final zero-result rate exceeds the available zero-result rate",
        "catalog zero-result rate exceeds the available zero-result rate",
    ]


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        (FakePipeline(retrieved={"milk": ["P1", "P1"]}), "retrieve returned duplicate"),
        (FakePipeline(retrieved={}, staged={("milk", "S1"): _both(["P1", "P1"])}), "search"),
    ],
)
def test_rejects_invalid_pipeline_output(pipeline: FakePipeline, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate(pipeline, JUDGMENTS, IN_STOCK)


def test_rejects_output_beyond_the_limit() -> None:
    class Greedy(FakePipeline):
        def retrieve(self, query: str, depth: int) -> Sequence[str]:
            return [f"P{i}" for i in range(depth + 1)]

    with pytest.raises(ValueError, match="returned 11 results for limit 10"):
        evaluate(Greedy(retrieved={}), JUDGMENTS, IN_STOCK, recall_at=(10,))


def test_needs_a_store() -> None:
    with pytest.raises(ValueError, match="at least one store"):
        evaluate(PIPELINE, JUDGMENTS, {})
