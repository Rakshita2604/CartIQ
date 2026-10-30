"""Compare two evaluation reports produced under identical conditions.

Reports are only comparable when everything except the pipeline is the same: snapshot,
catalog fingerprint, judgments, cut-offs, hit threshold and stores. Anything else would
attribute a change in the inputs to the pipeline.
"""

from collections.abc import Callable, Iterator

from cartiq.evaluation.runner import EvalReport, MetricSummary, QueryResult

_IDENTITY_FIELDS = (
    "snapshot_id",
    "catalog",
    "catalog_sha256",
    "judgments_sha256",
    "k",
    "recall_at",
    "hit_min_grade",
    "stores",
)


class IncomparableReportsError(ValueError):
    pass


def check_comparable(before: EvalReport, after: EvalReport) -> None:
    differences = [
        f"{name}: {getattr(before, name)!r} != {getattr(after, name)!r}"
        for name in _IDENTITY_FIELDS
        if getattr(before, name) != getattr(after, name)
    ]
    if differences:
        raise IncomparableReportsError("reports differ in " + "; ".join(differences))


def format_comparison(before: EvalReport, after: EvalReport, *, min_delta: float = 1e-9) -> str:
    """Group-level deltas, then every query whose NDCG, Recall or zero-result status changed."""
    check_comparable(before, after)
    depths = before.recall_at
    lines = [
        f"before={before.pipeline}  after={after.pipeline}  snapshot={before.snapshot_id}  "
        f"catalog={before.catalog_sha256[:12]}  judgments={before.judgments_sha256[:12]}  "
        f"stores={len(before.stores)}",
        "",
    ]
    lines += _group_table(before, after)
    lines += ["", f"Per-query NDCG@{before.k} changes (|delta| > 0):"]
    changes = sorted(
        _query_deltas(before, after, min_delta), key=lambda row: (row[0], row[1].query_id)
    )
    improved = [row for row in reversed(changes) if row[0] > 0]  # largest gain first
    regressed = [row for row in changes if row[0] < 0]  # largest loss first
    for title, rows in (("improved", improved), ("regressed", regressed)):
        lines.append(f"  {title} ({len(rows)}):")
        lines += [
            f"    {delta:+.3f}  {_fmt(b.ndcg)} -> {_fmt(a.ndcg)}  {a.segment:<11} {a.text!r}"
            for delta, a, b in rows
        ]
    lines += ["", f"Recall@{depths[-1]} and catalog zero-result changes:"]
    shifts = list(_coverage_shifts(before, after, depths[-1]))
    lines += shifts or ["  none"]
    return "\n".join(lines)


def _group_table(before: EvalReport, after: EvalReport) -> list[str]:
    depths = before.recall_at
    metrics: list[tuple[str, Callable[[MetricSummary], float | None]]] = [
        (f"ndcg@{before.k}", lambda s: s.ndcg),
        (f"mrr@{before.k}", lambda s: s.mrr),
        *((f"r@{d}", lambda s, d=d: s.recall[d]) for d in depths),
        ("zero%cat", lambda s: s.catalog_zero_rate * 100),
        ("zero%final", lambda s: s.final_zero_rate * 100),
    ]
    rows: list[tuple[str, MetricSummary, MetricSummary]] = [
        ("overall", before.overall, after.overall),
        *(
            (f"seg:{name}", summary, after.segments[name])
            for name, summary in before.segments.items()
        ),
        *(
            (f"attr:{name}", summary, after.attributes[name])
            for name, summary in before.attributes.items()
        ),
    ]
    width = max(len(name) for name, _, _ in rows)
    header = f"{'group':<{width}}" + "".join(f" {name:>22}" for name, _ in metrics)
    lines = [header, "-" * len(header)]
    for name, b, a in rows:
        cells = "".join(f" {_cell(metric(b), metric(a)):>22}" for _, metric in metrics)
        lines.append(f"{name:<{width}}{cells}")
    return lines


def _query_deltas(
    before: EvalReport, after: EvalReport, min_delta: float
) -> Iterator[tuple[float, QueryResult, QueryResult]]:
    for b, a in _paired(before, after):
        if a.ndcg is not None and b.ndcg is not None and abs(a.ndcg - b.ndcg) > min_delta:
            yield a.ndcg - b.ndcg, a, b


def _coverage_shifts(before: EvalReport, after: EvalReport, depth: int) -> Iterator[str]:
    for b, a in _paired(before, after):
        recall_b, recall_a = b.recall[depth], a.recall[depth]
        if recall_b != recall_a or b.catalog_zero != a.catalog_zero:
            yield (
                f"  {a.query_id:<28} r@{depth} {_fmt(recall_b)} -> {_fmt(recall_a)}  "
                f"catalog_zero {b.catalog_zero} -> {a.catalog_zero}"
            )


def _paired(before: EvalReport, after: EvalReport) -> Iterator[tuple[QueryResult, QueryResult]]:
    after_by_id = {query.query_id: query for query in after.queries}
    if set(after_by_id) != {query.query_id for query in before.queries}:
        raise IncomparableReportsError("reports cover different queries")
    for query in before.queries:
        yield query, after_by_id[query.query_id]


def _cell(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return f"{_fmt(before)} -> {_fmt(after)}"
    return f"{before:.3f}->{after:.3f} ({after - before:+.3f})"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
