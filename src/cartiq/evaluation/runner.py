"""Evaluate a search pipeline against a judgment set.

Two stages are measured separately:

- Retrieval, over the whole catalog and ignoring stock: Recall@K and the catalog
  zero-result rate.
- Final results, per store: NDCG@k (ideal built from judged products in stock at that
  store), MRR@k, Judged@k, and the available / final zero-result rates.

Per-store values are averaged into one value per query; segment, attribute and overall
figures average over queries, so every query weighs the same. Undefined values (see
``metrics``) are excluded and counted. Reports hold no timestamps: identical inputs give
identical reports.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from cartiq.domain import Coverage, Grade, JudgedQuery, QueryAttribute, QuerySegment
from cartiq.evaluation.judgments import JudgmentSet
from cartiq.evaluation.metrics import judged_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True, slots=True)
class StagedResults:
    available: Sequence[str]  # in-stock matches before any relaxation or substitution
    final: Sequence[str]  # what the shopper sees: exact, relaxed, then substitutes


class SearchPipeline(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, depth: int) -> Sequence[str]:
        """Up to ``depth`` candidates from the whole catalog, ignoring stock, best first."""
        ...

    def search(self, query: str, store_id: str, limit: int) -> StagedResults:
        """Results for one store; ``final`` holds at most ``limit`` products."""
        ...


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class RankedResult(_Frozen):
    product_id: str
    grade: Grade | None  # None: unjudged


class StoreResult(_Frozen):
    store_id: str
    ndcg: float | None
    reciprocal_rank: float | None
    judged: float | None
    available_zero: bool
    final_zero: bool
    results: list[RankedResult]


class QueryResult(_Frozen):
    query_id: str
    text: str
    segment: QuerySegment
    attributes: list[QueryAttribute]
    coverage: Coverage
    recall: dict[int, float | None]
    catalog_zero: bool
    ndcg: float | None
    reciprocal_rank: float | None
    judged: float | None
    available_zero_rate: float
    final_zero_rate: float
    unjudged_in_results: list[str]
    stores: list[StoreResult]


class MetricSummary(_Frozen):
    queries: int
    partial_queries: int  # > 0 marks every metric in this row as an estimate
    ndcg: float | None
    ndcg_undefined: int
    mrr: float | None
    mrr_undefined: int
    recall: dict[int, float | None]
    recall_undefined: int
    judged: float | None
    catalog_zero_rate: float
    available_zero_rate: float
    final_zero_rate: float


class EvalReport(_Frozen):
    pipeline: str
    snapshot_id: str | None
    catalog: str
    catalog_sha256: str
    labeler: str
    judgments_sha256: str
    k: int
    recall_at: list[int]
    hit_min_grade: Grade
    stores: list[str]
    overall: MetricSummary
    segments: dict[QuerySegment, MetricSummary]
    attributes: dict[QueryAttribute, MetricSummary]
    warnings: list[str]
    to_label: dict[str, list[str]]  # query id -> unjudged products that were shown
    queries: list[QueryResult]


def evaluate(
    pipeline: SearchPipeline,
    judgments: JudgmentSet,
    in_stock: Mapping[str, frozenset[str]],
    *,
    k: int = 10,
    recall_at: Sequence[int] = (10, 100),
    hit_min_grade: Grade = Grade.RELEVANT,
    snapshot_id: str | None = None,
) -> EvalReport:
    """``in_stock`` maps each store id to the ids of products in stock there."""
    if k < 1 or not recall_at or min(recall_at) < 1:
        raise ValueError("k and every recall_at depth must be >= 1")
    if not in_stock:
        raise ValueError("evaluation needs at least one store")
    depths = sorted(set(recall_at))
    stores = sorted(in_stock)
    results = [
        _evaluate_query(pipeline, query, in_stock, stores, k, depths, hit_min_grade)
        for query in judgments.queries
    ]

    overall = _summarize(results, depths)
    return EvalReport(
        pipeline=pipeline.name,
        snapshot_id=snapshot_id,
        catalog=judgments.catalog,
        catalog_sha256=judgments.catalog_sha256,
        labeler=judgments.labeler,
        judgments_sha256=judgments.sha256,
        k=k,
        recall_at=depths,
        hit_min_grade=hit_min_grade,
        stores=stores,
        overall=overall,
        # Groups in enum declaration order, so reports diff cleanly between runs.
        segments=_grouped(results, depths, QuerySegment, lambda r, s: r.segment is s),
        attributes=_grouped(results, depths, QueryAttribute, lambda r, a: a in r.attributes),
        warnings=_rate_warnings(overall),
        to_label={r.query_id: r.unjudged_in_results for r in results if r.unjudged_in_results},
        queries=results,
    )


def format_report(report: EvalReport) -> str:
    """Plain-text table. '~' marks rows that include partially judged queries (estimates)."""
    rows: list[tuple[str, MetricSummary]] = [
        *((f"seg:{name}", summary) for name, summary in report.segments.items()),
        *((f"attr:{name}", summary) for name, summary in report.attributes.items()),
        ("overall", report.overall),
    ]
    width = max(len(name) for name, _ in rows)
    recall_heads = "".join(f" {f'r@{depth}':>6}" for depth in report.recall_at)
    header = (
        f"{'group':<{width}} {'n':>3} {'part':>4}  {f'ndcg@{report.k}':>8} {f'mrr@{report.k}':>7}"
        f"{recall_heads} {'judged':>6}  {'zero% cat/avail/final':>21}  undef n/m/r"
    )
    lines = [
        f"pipeline={report.pipeline} snapshot={report.snapshot_id} catalog={report.catalog} "
        f"({report.catalog_sha256[:12]}) judgments={report.judgments_sha256[:12]} "
        f"stores={len(report.stores)} hit>={report.hit_min_grade.name}",
        header,
        "-" * len(header),
    ]
    for name, s in rows:
        mark = "~" if s.partial_queries else " "
        recalls = "".join(f" {_fmt(s.recall[depth]):>6}" for depth in report.recall_at)
        zeros = (
            f"{s.catalog_zero_rate * 100:5.1f}/{s.available_zero_rate * 100:5.1f}/"
            f"{s.final_zero_rate * 100:5.1f}"
        )
        lines.append(
            f"{name:<{width}} {s.queries:>3} {s.partial_queries:>4} {mark}{_fmt(s.ndcg):>8} "
            f"{_fmt(s.mrr):>7}{recalls} {_fmt(s.judged):>6}  {zeros:>21}  "
            f"{s.ndcg_undefined}/{s.mrr_undefined}/{s.recall_undefined}"
        )
    lines += [f"WARNING: {warning}" for warning in report.warnings]
    return "\n".join(lines)


def _evaluate_query(
    pipeline: SearchPipeline,
    query: JudgedQuery,
    in_stock: Mapping[str, frozenset[str]],
    stores: Sequence[str],
    k: int,
    depths: Sequence[int],
    hit_min_grade: Grade,
) -> QueryResult:
    candidates = _checked(pipeline.retrieve(query.text, depths[-1]), depths[-1], "retrieve")
    candidate_grades = [query.grade(pid) for pid in candidates]
    relevant = len(query.products_at_least(hit_min_grade))
    recall = {
        depth: recall_at_k(candidate_grades, depth, hit_min_grade, relevant) for depth in depths
    }

    store_results = [
        _evaluate_store(pipeline, query, store_id, in_stock[store_id], k, hit_min_grade)
        for store_id in stores
    ]
    unjudged = sorted({r.product_id for s in store_results for r in s.results if r.grade is None})
    return QueryResult(
        query_id=query.query_id,
        text=query.text,
        segment=query.segment,
        attributes=sorted(query.attributes, key=list(QueryAttribute).index),
        coverage=query.coverage,
        recall=recall,
        catalog_zero=not candidates,
        ndcg=_mean_defined(s.ndcg for s in store_results),
        reciprocal_rank=_mean_defined(s.reciprocal_rank for s in store_results),
        judged=_mean_defined(s.judged for s in store_results),
        available_zero_rate=fmean(float(s.available_zero) for s in store_results),
        final_zero_rate=fmean(float(s.final_zero) for s in store_results),
        unjudged_in_results=unjudged,
        stores=store_results,
    )


def _evaluate_store(
    pipeline: SearchPipeline,
    query: JudgedQuery,
    store_id: str,
    stocked: frozenset[str],
    k: int,
    hit_min_grade: Grade,
) -> StoreResult:
    staged = pipeline.search(query.text, store_id, k)
    final = _checked(staged.final, k, "search")
    grades = [query.grade(pid) for pid in final]
    ideal = [grade for pid, grade in query.grades.items() if pid in stocked and grade > 0]
    possible_hits = len(query.products_at_least(hit_min_grade) & stocked)
    return StoreResult(
        store_id=store_id,
        ndcg=ndcg_at_k(grades, ideal, k),
        reciprocal_rank=reciprocal_rank(grades, k, hit_min_grade, possible_hits),
        judged=judged_at_k(grades, k),
        available_zero=not staged.available,
        final_zero=not final,
        results=[
            RankedResult(product_id=pid, grade=g) for pid, g in zip(final, grades, strict=True)
        ],
    )


def _checked(ranking: Sequence[str], limit: int, stage: str) -> list[str]:
    ranking = list(ranking)
    if len(ranking) > limit:
        raise ValueError(f"{stage} returned {len(ranking)} results for limit {limit}")
    if len(set(ranking)) != len(ranking):
        raise ValueError(f"{stage} returned duplicate product ids")
    return ranking


def _grouped[G](
    results: Sequence[QueryResult],
    depths: Sequence[int],
    groups: Iterable[G],
    member: Callable[[QueryResult, G], bool],
) -> dict[G, MetricSummary]:
    summaries: dict[G, MetricSummary] = {}
    for group in groups:
        selected = [result for result in results if member(result, group)]
        if selected:
            summaries[group] = _summarize(selected, depths)
    return summaries


def _summarize(results: Sequence[QueryResult], depths: Sequence[int]) -> MetricSummary:
    deepest = depths[-1]
    return MetricSummary(
        queries=len(results),
        partial_queries=sum(r.coverage is Coverage.PARTIAL for r in results),
        ndcg=_mean_defined(r.ndcg for r in results),
        ndcg_undefined=sum(r.ndcg is None for r in results),
        mrr=_mean_defined(r.reciprocal_rank for r in results),
        mrr_undefined=sum(r.reciprocal_rank is None for r in results),
        recall={depth: _mean_defined(r.recall[depth] for r in results) for depth in depths},
        recall_undefined=sum(r.recall[deepest] is None for r in results),
        judged=_mean_defined(r.judged for r in results),
        catalog_zero_rate=fmean(float(r.catalog_zero) for r in results),
        available_zero_rate=fmean(r.available_zero_rate for r in results),
        final_zero_rate=fmean(r.final_zero_rate for r in results),
    )


def _rate_warnings(summary: MetricSummary) -> list[str]:
    """Check the expected ordering of zero-result *rates* (not result-list sizes).

    Fallbacks and substitutions only add results, so the final rate should not exceed the
    available rate; stock only removes results, so the catalog rate should not exceed it
    either. A violation points at a pipeline bug.
    """
    warnings = []
    if summary.final_zero_rate > summary.available_zero_rate:
        warnings.append("final zero-result rate exceeds the available zero-result rate")
    if summary.catalog_zero_rate > summary.available_zero_rate:
        warnings.append("catalog zero-result rate exceeds the available zero-result rate")
    return warnings


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return fmean(defined) if defined else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
