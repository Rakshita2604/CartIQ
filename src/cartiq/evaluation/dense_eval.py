"""Phase 5A evaluation of dense retrieval as a separate retriever.

For every (model, query template, product template) combination of a frozen bake-off plan:
standard metrics (the shared runner), complementarity against S0 and S6 candidates,
similarity distributions, and the model's query-encoding latency; then the pre-registered
selection rule. Development judgments only; reports hold no timestamps.

Complementarity semantics, kept separate:
- S0 reference: plain BM25F over the raw query, top ``reference_depth``.
- S6 reference: BM25F over the understood query concepts, top ``reference_depth``, before
  exact-first ordering (S6's candidate set).
Both use the same depth and the same relevance definitions (grade 3 = EXACT, grade 2+ =
RELEVANT or better). A dense-only candidate is a product in dense's top ``max(dense_depths)``
that the reference does not retrieve; every one is listed, with its grade (None when
unjudged) and the tier S6's constraints give it. A "useful" one is graded 2 or 3. Union
recall is the recall of reference candidates plus dense's top D, against the query's grade-3
and grade-2+ products separately. No similarity threshold is applied anywhere.
"""

import hashlib
import statistics
import time
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cartiq.domain import Grade, JudgedQuery, MatchTier, ScoredProduct
from cartiq.evaluation.pipelines import DensePipeline
from cartiq.evaluation.runner import EvalReport
from cartiq.retrieval.dense import DenseHit


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlanMetrics(_Strict):
    k: int
    recall_at: list[int]
    hit_min_grade: int


class PlanComplementarity(_Strict):
    reference_depth: int
    dense_depths: list[int]
    misleading_depth: int


class PlanSimilarity(_Strict):
    top_ranks: list[int]


class PlanSelection(_Strict):
    primary: Literal["recall@20"]
    secondary: Literal["ndcg@10"]
    latency_p95_ms: float
    model_size_order: list[str]


class BakeoffPlan(_Strict):
    schema_version: Literal[1]
    version: str
    models: list[str]
    query_templates: list[str]
    product_templates: list[str]
    judgments: str
    catalog: str
    metrics: PlanMetrics
    complementarity: PlanComplementarity
    similarity: PlanSimilarity
    selection: PlanSelection


def load_plan(path: Path) -> tuple[BakeoffPlan, str]:
    raw = path.read_bytes()
    plan = BakeoffPlan.model_validate(tomllib.loads(raw.decode()))
    if "heldout" in Path(plan.judgments).name:
        raise ValueError("the bake-off runs on development judgments only, never held-out")
    return plan, hashlib.sha256(raw).hexdigest()


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Candidate(_Frozen):
    query: str
    product: str
    grade: int
    dense_rank: int


class DenseOnly(_Frozen):
    query: str
    product: str
    dense_rank: int
    grade: int | None  # None: unjudged
    s6_tier: str  # the tier S6's constraints give it: "exact" or "relaxed"
    s6_relaxed: list[str]  # the constraints it violates


class UnionRecall(_Frozen):
    reference_grade3: float | None
    union_grade3: float | None
    reference_grade2plus: float | None
    union_grade2plus: float | None


class Complementarity(_Frozen):
    reference: str
    useful_missed_grade3: list[Candidate]
    useful_missed_grade2: list[Candidate]
    useful_s6_exact: int  # useful candidates S6's constraints would make EXACT
    dense_only: list[DenseOnly]
    queries_with_additions: int
    queries_without_additions: list[str]
    union_recall: dict[int, UnionRecall]  # dense depth -> recall


class Distribution(_Frozen):
    n: int
    minimum: float
    median: float
    maximum: float
    mean: float


class Similarity(_Frozen):
    # "with_relevant" / "without_relevant", and per segment; rank -> distribution
    by_relevance: dict[str, dict[int, Distribution]]
    by_segment: dict[str, dict[int, Distribution]]


class CombinationResult(_Frozen):
    model: str
    query_template: str
    product_template: str
    pipeline: str
    artifact_id: str
    encoder: str  # repository, revision, ONNX and tokenizer SHA-256, pooling, prefixes
    catalog_sha256: str
    vectors_sha256: str
    overall: dict[str, float | None]
    segments: dict[str, dict[str, float | None]]
    complementarity: list[Complementarity]
    misleading: list[Candidate]
    similarity: Similarity


class LatencyResult(_Frozen):
    model: str
    queries: int
    p50_ms: float
    p95_ms: float
    eligible: bool


class BakeoffReport(_Frozen):
    plan_version: str
    plan_sha256: str
    snapshot_id: str
    judgments: str
    judgments_sha256: str
    disclosure: str
    latency: list[LatencyResult]
    combinations: list[CombinationResult]
    ranking: list[str]  # pipeline names, best first by the pre-registered rule
    selected: str | None


DISCLOSURE = (
    "Model selection on development data: the 63 development queries were seen during "
    "Phases 2-4, so this is not an independent test. Held-out judgments were not used."
)


def combination_result(
    plan: BakeoffPlan,
    pipeline: DensePipeline,
    report: EvalReport,
    queries: Sequence[JudgedQuery],
    references: Mapping[str, Mapping[str, Sequence[str]]],
    classify: Callable[[str, Sequence[str]], Mapping[str, ScoredProduct]],
    *,
    model: str,
    query_template: str,
    product_template: str,
    artifact_id: str,
) -> CombinationResult:
    """``references`` maps "S0"/"S6" to {query text: top reference_depth product ids};
    ``classify`` gives S6's constraint tier for products of a query."""
    depth = max(
        plan.complementarity.dense_depths
        + plan.similarity.top_ranks
        + [plan.complementarity.misleading_depth]
    )
    dense = {q.text: pipeline.hits(q.text, depth) for q in queries}
    return CombinationResult(
        model=model,
        query_template=query_template,
        product_template=product_template,
        pipeline=pipeline.name,
        artifact_id=artifact_id,
        encoder=pipeline.encoder_identity,
        catalog_sha256=pipeline.catalog_sha256,
        vectors_sha256=pipeline.vectors_sha256,
        overall=_summary(report.overall.model_dump()),
        segments={s.value: _summary(v.model_dump()) for s, v in report.segments.items()},
        complementarity=[
            _complementarity(name, plan, queries, dense, ranked, classify)
            for name, ranked in references.items()
        ],
        misleading=[
            Candidate(query=q.text, product=hit.product_id, grade=0, dense_rank=hit.rank)
            for q in queries
            for hit in dense[q.text][: plan.complementarity.misleading_depth]
            if q.grade(hit.product_id) == Grade.IRRELEVANT
        ],
        similarity=_similarity(plan, queries, dense),
    )


def encoding_latency(
    model: str,
    encode: Callable[[str], object],
    texts: Sequence[str],
    limit_ms: float,
    repeat: int = 3,
) -> LatencyResult:
    """Single-query encoding latency (one text per call) after one warm-up pass."""
    for text in texts:
        encode(text)
    samples = []
    for _ in range(repeat):
        for text in texts:
            started = time.perf_counter()
            encode(text)
            samples.append((time.perf_counter() - started) * 1000)
    p95 = statistics.quantiles(samples, n=100, method="inclusive")[94]
    return LatencyResult(
        model=model,
        queries=len(samples),
        p50_ms=round(statistics.median(samples), 3),
        p95_ms=round(p95, 3),
        eligible=p95 <= limit_ms,
    )


def select(
    plan: BakeoffPlan, combinations: Sequence[CombinationResult], latency: Sequence[LatencyResult]
) -> list[str]:
    """Pipeline names, best first, by the pre-registered rule (ineligible ones excluded)."""
    eligible = {result.model for result in latency if result.eligible}
    order = plan.selection.model_size_order
    ranked = sorted(
        (c for c in combinations if c.model in eligible),
        key=lambda c: (
            -(c.overall["recall@20"] or 0.0),
            -(c.overall["ndcg@10"] or 0.0),
            order.index(c.model) if c.model in order else len(order),
            c.pipeline,
        ),
    )
    return [c.pipeline for c in ranked]


def _summary(summary: Mapping[str, object]) -> dict[str, float | None]:
    recall = summary["recall"]
    assert isinstance(recall, Mapping)
    return {
        "ndcg@10": _float(summary["ndcg"]),
        "mrr@10": _float(summary["mrr"]),
        **{f"recall@{depth}": _float(value) for depth, value in sorted(recall.items())},
        "queries": _float(summary["queries"]),
    }


def _float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return round(float(value), 6)


def _complementarity(
    name: str,
    plan: BakeoffPlan,
    queries: Sequence[JudgedQuery],
    dense: Mapping[str, Sequence[DenseHit]],
    reference: Mapping[str, Sequence[str]],
    classify: Callable[[str, Sequence[str]], Mapping[str, ScoredProduct]],
) -> Complementarity:
    deepest = max(plan.complementarity.dense_depths)
    grade3: list[Candidate] = []
    grade2: list[Candidate] = []
    dense_only: list[DenseOnly] = []
    useful_exact = 0
    without: list[str] = []
    union: dict[int, list[tuple[float | None, float | None, float | None, float | None]]] = (
        defaultdict(list)
    )
    for query in queries:
        ref = set(reference[query.text][: plan.complementarity.reference_depth])
        hits = dense[query.text]
        added = False
        only = [hit for hit in hits[:deepest] if hit.product_id not in ref]
        tiers = classify(query.text, [hit.product_id for hit in only])
        for hit in only:
            grade = query.grade(hit.product_id)
            tier = tiers[hit.product_id]
            dense_only.append(
                DenseOnly(
                    query=query.text,
                    product=hit.product_id,
                    dense_rank=hit.rank,
                    grade=None if grade is None else int(grade),
                    s6_tier=tier.tier.value,
                    s6_relaxed=sorted(c.value for c in tier.relaxed),
                )
            )
            if grade is not None and grade >= Grade.RELEVANT:
                item = Candidate(
                    query=query.text, product=hit.product_id, grade=int(grade), dense_rank=hit.rank
                )
                (grade3 if grade is Grade.EXACT else grade2).append(item)
                useful_exact += tier.tier is MatchTier.EXACT
                added = True
        if not added:
            without.append(query.text)
        exact = query.products_at_least(Grade.EXACT)
        relevant = query.products_at_least(Grade.RELEVANT)
        for depth in plan.complementarity.dense_depths:
            both = ref | {hit.product_id for hit in hits[:depth]}
            union[depth].append(
                (
                    _recall(ref, exact),
                    _recall(both, exact),
                    _recall(ref, relevant),
                    _recall(both, relevant),
                )
            )
    return Complementarity(
        reference=name,
        useful_missed_grade3=grade3,
        useful_missed_grade2=grade2,
        useful_s6_exact=useful_exact,
        dense_only=dense_only,
        queries_with_additions=len(queries) - len(without),
        queries_without_additions=without,
        union_recall={
            depth: UnionRecall(
                reference_grade3=_mean([r[0] for r in rows]),
                union_grade3=_mean([r[1] for r in rows]),
                reference_grade2plus=_mean([r[2] for r in rows]),
                union_grade2plus=_mean([r[3] for r in rows]),
            )
            for depth, rows in union.items()
        },
    )


def _recall(found: set[str], targets: frozenset[str]) -> float | None:
    return len(found & targets) / len(targets) if targets else None


def _mean(values: Sequence[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return round(statistics.fmean(defined), 6) if defined else None


def _similarity(
    plan: BakeoffPlan, queries: Sequence[JudgedQuery], dense: Mapping[str, Sequence[DenseHit]]
) -> Similarity:
    by_relevance: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_segment: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for query in queries:
        group = "with_relevant" if query.products_at_least(Grade.RELEVANT) else "without_relevant"
        hits = dense[query.text]
        for rank in plan.similarity.top_ranks:
            if len(hits) >= rank:
                score = hits[rank - 1].score
                by_relevance[group][rank].append(score)
                by_segment[query.segment.value][rank].append(score)
    return Similarity(
        by_relevance={
            g: {r: _distribution(v) for r, v in ranks.items()}
            for g, ranks in sorted(by_relevance.items())
        },
        by_segment={
            g: {r: _distribution(v) for r, v in ranks.items()}
            for g, ranks in sorted(by_segment.items())
        },
    )


def _distribution(values: Sequence[float]) -> Distribution:
    return Distribution(
        n=len(values),
        minimum=round(min(values), 6),
        median=round(statistics.median(values), 6),
        maximum=round(max(values), 6),
        mean=round(statistics.fmean(values), 6),
    )
