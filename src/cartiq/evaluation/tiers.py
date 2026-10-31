"""How well constraint-first ordering's EXACT tier matches the hand labels (stage s6).

For each development query, at catalog level (no stock), within the ``candidate_depth``
best BM25F candidates:

- exact precision: EXACT-tier products that are labelled grade 3 / EXACT-tier products;
- exact recall: grade-3 products in the EXACT tier / grade-3 products.

The grade-3 labels were written independently of the constraint logic. Every mismatch is
listed with its reason. Without labels for the catalog (e.g. BigBasket), only the counts
are reported. Reports hold no timestamps.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from cartiq.domain import Grade, JudgedQuery, Product
from cartiq.evaluation.pipelines import UnderstandingPipeline


@dataclass(frozen=True, slots=True)
class TierQuery:
    query_id: str
    text: str
    judged: JudgedQuery | None  # None: no labels for this catalog


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class QueryTiers(_Frozen):
    query_id: str
    text: str
    brand: str | None
    quantity: str | None
    candidates: int
    eligible_matches: int
    exact: int
    relaxed: int
    relaxed_by: dict[str, int]  # violation -> RELAXED results in the top k carrying it
    exact_beyond_depth: int  # EXACT products past candidate_depth (cannot be promoted)
    top: list[str]  # top k as "tier:product name"
    exact_precision: float | None = None
    exact_recall: float | None = None
    false_exact: list[str] = []  # EXACT but not grade 3
    missed_exact: list[str] = []  # grade 3 but not EXACT, with the reason


class TiersReport(_Frozen):
    pipeline: str
    snapshot_id: str
    labelled: bool
    judgments_sha256: str | None
    candidate_depth: int
    k: int
    queries: int
    queries_with_exact: int
    exact_true_positives: int | None
    exact_false_positives: int | None
    exact_false_negatives: int | None
    exact_precision: float | None
    exact_recall: float | None
    relaxed_by: dict[str, int]
    exact_beyond_depth: int
    query_results: list[QueryTiers]


def evaluate_tiers(
    pipeline: UnderstandingPipeline,
    queries: Sequence[TierQuery],
    products: Mapping[str, Product],
    *,
    snapshot_id: str,
    judgments_sha256: str | None,
    candidate_depth: int,
    k: int,
) -> TiersReport:
    results = [_query_tiers(pipeline, query, products, k) for query in queries]
    labelled = all(query.judged is not None for query in queries)
    tp = sum(r.exact - len(r.false_exact) for r in results) if labelled else None
    fp = sum(len(r.false_exact) for r in results) if labelled else None
    fn = sum(len(r.missed_exact) for r in results) if labelled else None
    relaxed_by: Counter[str] = Counter()
    for r in results:
        relaxed_by.update(r.relaxed_by)
    return TiersReport(
        pipeline=pipeline.name,
        snapshot_id=snapshot_id,
        labelled=labelled,
        judgments_sha256=judgments_sha256 if labelled else None,
        candidate_depth=candidate_depth,
        k=k,
        queries=len(results),
        queries_with_exact=sum(r.exact > 0 for r in results),
        exact_true_positives=tp,
        exact_false_positives=fp,
        exact_false_negatives=fn,
        exact_precision=_ratio(tp, (tp or 0) + (fp or 0)) if labelled else None,
        exact_recall=_ratio(tp, (tp or 0) + (fn or 0)) if labelled else None,
        relaxed_by=dict(sorted(relaxed_by.items())),
        exact_beyond_depth=sum(r.exact_beyond_depth for r in results),
        query_results=results,
    )


def format_tiers_report(report: TiersReport) -> str:
    def ratio(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    lines = [
        f"pipeline={report.pipeline} snapshot={report.snapshot_id} "
        f"candidate_depth={report.candidate_depth} k={report.k} labelled={report.labelled}",
        f"queries with an EXACT product: {report.queries_with_exact}/{report.queries}; "
        f"EXACT products beyond depth: {report.exact_beyond_depth}",
        f"RELAXED results in top {report.k} by violation: {report.relaxed_by or 'none'}",
    ]
    if report.labelled:
        lines.append(
            f"exact vs grade 3: precision {ratio(report.exact_precision)} recall "
            f"{ratio(report.exact_recall)} (tp {report.exact_true_positives}, fp "
            f"{report.exact_false_positives}, fn {report.exact_false_negatives})"
        )
    for r in report.query_results:
        if r.false_exact or r.missed_exact or (report.labelled is False and r.exact == 0):
            lines.append(
                f"  {r.text!r}: exact {r.exact}, relaxed {r.relaxed}, candidates {r.candidates}"
                f"/{r.eligible_matches}"
            )
            lines += [f"      false exact: {item}" for item in r.false_exact]
            lines += [f"      missed exact: {item}" for item in r.missed_exact]
    return "\n".join(lines)


def _query_tiers(
    pipeline: UnderstandingPipeline, query: TierQuery, products: Mapping[str, Product], k: int
) -> QueryTiers:
    parsed = pipeline.parse(query.text)
    ordered = pipeline.ordered(query.text, count_matches=True)
    exact = ordered.exact
    tier_of = {s.product_id: s for s in ordered.scored}
    relaxed_by: Counter[str] = Counter()
    for scored in ordered.scored[:k]:
        relaxed_by.update(constraint.value for constraint in scored.relaxed)
    mention = parsed.quantity_mention
    precision = recall = None
    false_exact: list[str] = []
    missed_exact: list[str] = []
    if query.judged is not None:
        judged = query.judged
        grade3 = judged.products_at_least(Grade.EXACT)
        hits = [pid for pid in exact if pid in grade3]
        precision = _ratio(len(hits), len(exact))
        recall = _ratio(len(hits), len(grade3))
        false_exact = [
            f"{products[pid].name} (grade {int(judged.grade(pid) or 0)})"
            for pid in exact
            if pid not in grade3
        ]
        for pid in sorted(grade3 - set(exact)):
            if pid in tier_of:
                reason = (
                    "relaxed (" + ", ".join(sorted(c.value for c in tier_of[pid].relaxed)) + ")"
                )
            else:
                reason = f"not among the top {len(ordered.scored)} candidates"
            missed_exact.append(f"{products[pid].name}: {reason}")
    return QueryTiers(
        query_id=query.query_id,
        text=query.text,
        brand=parsed.brand,
        quantity=mention.text if mention else None,
        candidates=len(ordered.scored),
        eligible_matches=ordered.eligible_matches or 0,
        exact=len(exact),
        relaxed=len(ordered.relaxed),
        relaxed_by=dict(sorted(relaxed_by.items())),
        exact_beyond_depth=pipeline.beyond_depth_exact(query.text),
        top=[f"{s.tier.value}:{products[s.product_id].name}" for s in ordered.scored[:k]],
        exact_precision=precision,
        exact_recall=recall,
        false_exact=false_exact,
        missed_exact=missed_exact,
    )


def _ratio(numerator: int | None, denominator: int) -> float | None:
    if numerator is None or denominator == 0:
        return None
    return numerator / denominator
