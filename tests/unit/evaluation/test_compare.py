import pytest

from cartiq.domain import Coverage, Grade, QuerySegment
from cartiq.evaluation.compare import IncomparableReportsError, check_comparable, format_comparison
from cartiq.evaluation.runner import EvalReport, MetricSummary, QueryResult


def _summary(ndcg: float, recall: float, zero: float) -> MetricSummary:
    return MetricSummary(
        queries=2,
        partial_queries=0,
        ndcg=ndcg,
        ndcg_undefined=0,
        mrr=ndcg,
        mrr_undefined=0,
        recall={10: recall, 100: recall},
        recall_undefined=0,
        judged=1.0,
        catalog_zero_rate=zero,
        available_zero_rate=zero,
        final_zero_rate=zero,
    )


def _query(query_id: str, ndcg: float | None, recall: float, catalog_zero: bool) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        text=query_id,
        segment=QuerySegment.HEAD,
        attributes=[],
        coverage=Coverage.EXHAUSTIVE,
        recall={10: recall, 100: recall},
        catalog_zero=catalog_zero,
        ndcg=ndcg,
        reciprocal_rank=ndcg,
        judged=1.0,
        available_zero_rate=float(catalog_zero),
        final_zero_rate=float(catalog_zero),
        unjudged_in_results=[],
        stores=[],
    )


def _report(pipeline: str, queries: list[QueryResult], summary: MetricSummary) -> EvalReport:
    return EvalReport(
        pipeline=pipeline,
        snapshot_id="snap-x",
        catalog="cat",
        catalog_sha256="c" * 64,
        labeler="t",
        judgments_sha256="j" * 64,
        k=10,
        recall_at=[10, 100],
        hit_min_grade=Grade.RELEVANT,
        stores=["S1"],
        overall=summary,
        segments={QuerySegment.HEAD: summary},
        attributes={},
        warnings=[],
        to_label={},
        queries=queries,
    )


BEFORE = _report(
    "base",
    [
        _query("up", 0.2, 1.0, False),
        _query("down", 0.9, 1.0, False),
        _query("lost", 0.5, 1.0, False),
    ],
    _summary(0.5, 1.0, 0.0),
)
AFTER = _report(
    "new",
    [
        _query("up", 0.8, 1.0, False),
        _query("down", 0.7, 1.0, False),
        _query("lost", 0.0, 0.0, True),
    ],
    _summary(0.5, 2 / 3, 1 / 3),
)


def test_lists_improvements_regressions_and_coverage_losses() -> None:
    text = format_comparison(BEFORE, AFTER)

    assert "0.500->0.500 (+0.000)" in text  # overall NDCG
    assert "1.000->0.667 (-0.333)" in text  # overall recall
    improved = text.split("improved (1):")[1].split("regressed")[0]
    assert "+0.600  0.200 -> 0.800" in improved
    regressed = text.split("regressed (2):")[1].split("Recall@100")[0]
    # Largest loss first.
    assert regressed.index("'lost'") < regressed.index("'down'")
    assert "lost                         r@100 1.000 -> 0.000  catalog_zero False -> True" in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snapshot_id", "snap-y"),
        ("catalog_sha256", "d" * 64),
        ("judgments_sha256", "k" * 64),
        ("k", 5),
        ("recall_at", [10]),
        ("hit_min_grade", Grade.EXACT),
        ("stores", ["S1", "S2"]),
    ],
)
def test_refuses_reports_from_different_conditions(field: str, value: object) -> None:
    other = AFTER.model_copy(update={field: value})

    with pytest.raises(IncomparableReportsError, match=field):
        check_comparable(BEFORE, other)


def test_refuses_different_query_sets() -> None:
    other = AFTER.model_copy(update={"queries": AFTER.queries[:2]})

    with pytest.raises(IncomparableReportsError, match="different queries"):
        format_comparison(BEFORE, other)
