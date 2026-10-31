"""Command-line entry point for the API server and offline jobs (snapshots, evaluation)."""

import contextlib
import hashlib
import json
import logging
import statistics
import time
import tracemalloc
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from cartiq.catalog_fingerprint import catalog_fingerprint
from cartiq.clock import SystemClock
from cartiq.config import Settings
from cartiq.domain import Grade, Product
from cartiq.evaluation import (
    JudgmentError,
    check_against_catalog,
    evaluate,
    format_report,
    load_judgments,
)
from cartiq.evaluation.compare import IncomparableReportsError, format_comparison
from cartiq.evaluation.pipelines import BM25Pipeline, TokenOverlapPipeline, UnderstandingPipeline
from cartiq.evaluation.qu_eval import evaluate_suite, format_suite_report, load_expectations
from cartiq.evaluation.runner import EvalReport
from cartiq.evaluation.tiers import TierQuery, evaluate_tiers, format_tiers_report
from cartiq.ingestion.bigbasket import read_bigbasket_csv
from cartiq.ingestion.raw import RawCatalogRow
from cartiq.ingestion.synthetic import SYNTHETIC_CATALOG_VERSION, synthetic_catalog_rows
from cartiq.parsing.text import tokenize
from cartiq.query import STAGES
from cartiq.query.factory import build_parser, parser_identity
from cartiq.query.spelling import spelling_frequencies
from cartiq.retrieval import BM25FIndex, bm25_config_id
from cartiq.snapshots import (
    DB_FILENAME,
    SnapshotManifest,
    SnapshotVerificationError,
    SourceInfo,
    build_snapshot,
    load_manifest,
    rows_sha256,
    snapshots_root,
    source_sha256,
    verify_snapshot,
)
from cartiq.storage.sqlite import connect, sqlite_repositories

app = typer.Typer(no_args_is_help=True, add_completion=False)


class CatalogSource(StrEnum):
    SYNTHETIC = "synthetic"
    BIGBASKET = "bigbasket"


class RankerChoice(StrEnum):
    BASELINE = "baseline"
    BM25 = "bm25"
    QU = "qu"  # BM25F over query understanding; pick the stage with --stage


class StageChoice(StrEnum):
    S1 = "s1"  # + quantity extraction
    S2 = "s2"  # + Hinglish lexicon
    S3 = "s3"  # + spelling correction
    S4 = "s4"  # + plural variants
    S5 = "s5"  # + brand and alias recognition, spelling protection
    S6 = "s6"  # + constraint-first ordering (EXACT before RELAXED)


@app.command()
def serve(
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev only).")] = False,
) -> None:
    """Run the search API."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(
        "cartiq.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.lower(),
        reload=reload,
    )


@app.command("show-config")
def show_config() -> None:
    """Print the effective configuration after file and environment overrides."""
    typer.echo(Settings().model_dump_json(indent=2))


@app.command("build-snapshot")
def build_snapshot_command(
    source: Annotated[
        CatalogSource, typer.Option(help="Catalog source.")
    ] = CatalogSource.SYNTHETIC,
    input_path: Annotated[
        Path | None,
        typer.Option("--input", help="BigBasket CSV path (defaults to ingestion.bigbasket_csv)."),
    ] = None,
) -> None:
    """Ingest a catalog, simulate stores/inventory/users/orders, and write a snapshot."""
    settings = Settings()
    rows: list[RawCatalogRow]
    if source is CatalogSource.SYNTHETIC:
        rows = synthetic_catalog_rows()
        info = SourceInfo(
            kind="synthetic", name=SYNTHETIC_CATALOG_VERSION, sha256=rows_sha256(rows)
        )
    else:
        path = input_path or settings.ingestion.bigbasket_csv
        if not path.is_file():
            raise typer.BadParameter(
                f"{path} not found. Download 'BigBasket Products.csv' from Kaggle "
                "(surajjha101/bigbasket-entire-product-list-28k-datapoints, CC BY-NC-SA 4.0).",
                param_hint="--input",
            )
        rows = list(read_bigbasket_csv(path))
        info = SourceInfo(kind="bigbasket", name=path.name, sha256=source_sha256(path))

    manifest = build_snapshot(
        rows, info, settings.simulation, settings.paths.artifacts_dir, SystemClock()
    )
    typer.echo(f"snapshot: {manifest.snapshot_id}")
    typer.echo(f"path:     {snapshots_root(settings.paths.artifacts_dir) / manifest.snapshot_id}")
    typer.echo(f"catalog:  {manifest.catalog_sha256}")
    typer.echo(f"counts:   {manifest.counts}")
    typer.echo(f"ingest:   {manifest.ingestion_report}")


@app.command("verify-snapshot")
def verify_snapshot_command(snapshot_id: str) -> None:
    """Recompute a snapshot's content hash and compare it with its manifest."""
    snapshot_dir = snapshots_root(Settings().paths.artifacts_dir) / snapshot_id
    if not snapshot_dir.is_dir():
        raise typer.BadParameter(f"no snapshot at {snapshot_dir}", param_hint="SNAPSHOT_ID")
    try:
        manifest = verify_snapshot(snapshot_dir)
    except SnapshotVerificationError as error:
        typer.echo(f"FAILED: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"OK: {manifest.snapshot_id} ({manifest.content_sha256})")
    if manifest.catalog_sha256:
        typer.echo(f"catalog fingerprint: {manifest.catalog_sha256}")


@app.command("evaluate")
def evaluate_command(
    snapshot_id: str,
    ranker: Annotated[RankerChoice, typer.Option(help="Ranker to evaluate.")] = (
        RankerChoice.BASELINE
    ),
    judgments_path: Annotated[
        Path | None,
        typer.Option("--judgments", help="Judgment file (defaults to evaluation.judgments_file)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Report JSON path (defaults to artifacts/eval/<snapshot>/...)."),
    ] = None,
    stage: Annotated[
        StageChoice, typer.Option(help="Query-understanding stage for --ranker qu.")
    ] = StageChoice.S4,
) -> None:
    """Score a pipeline against hand-labeled judgments (see eval/README.md)."""
    settings = Settings()
    manifest, products, in_stock = _load_snapshot(settings, snapshot_id)
    try:
        judgments = load_judgments(judgments_path or settings.evaluation.judgments_file)
        check_against_catalog(
            judgments,
            catalog=manifest.source.name,
            fingerprint=catalog_fingerprint(products),
            product_ids={product.product_id for product in products},
        )
    except JudgmentError as error:
        typer.echo(f"FAILED: {error}", err=True)
        raise typer.Exit(1) from error

    pipeline: TokenOverlapPipeline | BM25Pipeline | UnderstandingPipeline
    if ranker is RankerChoice.BM25:
        pipeline = BM25Pipeline(products, in_stock, settings.retrieval.bm25)
    elif ranker is RankerChoice.QU:
        pipeline = UnderstandingPipeline(
            products,
            in_stock,
            settings.retrieval.bm25,
            settings.query,
            stage.value,
            STAGES[stage.value],
            settings.ranking,
        )
    else:
        pipeline = TokenOverlapPipeline(products, in_stock)
    report = evaluate(
        pipeline,
        judgments,
        in_stock,
        k=settings.evaluation.k,
        recall_at=settings.evaluation.recall_at,
        hit_min_grade=Grade(settings.evaluation.hit_min_grade),
        snapshot_id=snapshot_id,
    )
    path = output or (
        settings.paths.artifacts_dir
        / "eval"
        / snapshot_id
        / f"{report.pipeline}-{report.judgments_sha256[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")
    typer.echo(format_report(report))
    typer.echo(f"report: {path}")


@app.command("compare-reports")
def compare_reports_command(before: Path, after: Path) -> None:
    """Compare two evaluation reports made on the same snapshot, judgments and settings."""
    reports = [EvalReport.model_validate_json(path.read_text()) for path in (before, after)]
    try:
        typer.echo(format_comparison(*reports))
    except IncomparableReportsError as error:
        typer.echo(f"FAILED: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("bench-bm25")
def bench_bm25_command(
    snapshot_id: str,
    repeat: Annotated[int, typer.Option(min=1, help="Timed passes over the workload.")] = 3,
) -> None:
    """Time BM25F index build and queries. A benchmark on this machine, not a guarantee.

    The workload is the query texts of the judgment file, run against the whole catalog
    (retrieval depth) and against every store's stock (result limit).
    """
    settings = Settings()
    _, products, in_stock = _load_snapshot(settings, snapshot_id)
    queries = [query.text for query in load_judgments(settings.evaluation.judgments_file).queries]
    depth, limit = max(settings.evaluation.recall_at), settings.evaluation.k

    started = time.perf_counter()
    index = BM25FIndex.build(products, settings.retrieval.bm25)
    build_ms = (time.perf_counter() - started) * 1000

    def timed_search(query: str, size: int, allowed: frozenset[str] | None = None) -> float:
        started = time.perf_counter()
        index.search(query, size, allowed=allowed)
        return (time.perf_counter() - started) * 1000

    for query in queries:  # warm-up pass, not timed
        index.search(query, depth)
    catalog_ms: list[float] = []
    store_ms: list[float] = []
    for _ in range(repeat):
        for query in queries:
            catalog_ms.append(timed_search(query, depth))
            store_ms += [timed_search(query, limit, stocked) for stocked in in_stock.values()]

    typer.echo(
        f"snapshot {snapshot_id}: {len(index)} products, {index.term_count} terms, "
        f"config {bm25_config_id(settings.retrieval.bm25)}"
    )
    typer.echo(f"index build: {build_ms:.0f} ms")
    for label, samples in (
        (f"catalog search (depth {depth})", catalog_ms),
        (f"store search (limit {limit})", store_ms),
    ):
        p50 = statistics.median(samples)
        p95 = statistics.quantiles(samples, n=100, method="inclusive")[94]
        typer.echo(
            f"{label}: n={len(samples)} p50={p50:.2f} ms p95={p95:.2f} ms max={max(samples):.2f} ms"
        )


def _load_snapshot(
    settings: Settings, snapshot_id: str
) -> tuple[SnapshotManifest, list[Product], dict[str, frozenset[str]]]:
    snapshot_dir = snapshots_root(settings.paths.artifacts_dir) / snapshot_id
    if not snapshot_dir.is_dir():
        raise typer.BadParameter(f"no snapshot at {snapshot_dir}", param_hint="SNAPSHOT_ID")
    with contextlib.closing(connect(snapshot_dir / DB_FILENAME, read_only=True)) as conn:
        repos = sqlite_repositories(conn)
        products = repos.catalog.list_all()
        in_stock = {
            store.store_id: repos.stock.in_stock_ids(store.store_id)
            for store in repos.stores.list_all()
        }
    return load_manifest(snapshot_dir), products, in_stock


@app.command("bench-query")
def bench_query_command(
    snapshot_id: str,
    stage: Annotated[StageChoice, typer.Option(help="Query-understanding stage.")] = (
        StageChoice.S4
    ),
    repeat: Annotated[int, typer.Option(min=1, help="Timed passes over the workload.")] = 3,
) -> None:
    """Time query understanding on a snapshot and count zero-result queries (no labels).

    The workload is the development judgment file's query texts, never the held-out ones.
    A benchmark on this machine, not a guarantee.
    """
    settings = Settings()
    _, products, in_stock = _load_snapshot(settings, snapshot_id)
    queries = [query.text for query in load_judgments(settings.evaluation.judgments_file).queries]
    depth, limit = max(settings.evaluation.recall_at), settings.evaluation.k

    index = BM25FIndex.build(products, settings.retrieval.bm25)
    spelling_df = spelling_frequencies(products, settings.query.spelling_fields)
    stages = STAGES[stage.value]
    started = time.perf_counter()
    parser = build_parser(products, index.vocabulary, settings.query, stages)
    build_ms = (time.perf_counter() - started) * 1000
    tracemalloc.start()
    build_parser(products, index.vocabulary, settings.query, stages)
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()

    def timed(action: Callable[[], object]) -> float:
        started = time.perf_counter()
        action()
        return (time.perf_counter() - started) * 1000

    parse_ms: list[float] = []
    catalog_ms: list[float] = []
    store_ms: list[float] = []
    for _ in range(repeat):
        for query in queries:
            parse_ms.append(timed(partial(parser.parse, query)))
            concepts = parser.parse(query).concepts
            catalog_ms.append(timed(partial(index.search_concepts, concepts, depth)))
            store_ms += [
                timed(partial(index.search_concepts, concepts, limit, stocked))
                for stocked in in_stock.values()
            ]
    if stages.ordering:
        pipeline = UnderstandingPipeline(
            products,
            in_stock,
            settings.retrieval.bm25,
            settings.query,
            stage.value,
            stages,
            settings.ranking,
        )
        for query in queries:  # warm the parse cache: time ordering, not parsing
            pipeline.parse(query)
        ordered_ms = [
            timed(partial(pipeline.search, query, store_id, limit))
            for _ in range(repeat)
            for query in queries
            for store_id in in_stock
        ]
        p50 = statistics.median(ordered_ms)
        p95 = statistics.quantiles(ordered_ms, n=100, method="inclusive")[94]
        typer.echo(
            f"store search with ordering (depth {settings.ranking.candidate_depth} -> limit "
            f"{limit}): n={len(ordered_ms)} p50={p50:.3f} ms p95={p95:.3f} ms"
        )
    plain_zero = sum(not index.search(query, depth) for query in queries)
    stage_zero = sum(not index.search_concepts(parser.parse(q).concepts, depth) for q in queries)

    typer.echo(f"snapshot {snapshot_id}: {len(index)} products, stage {stage.value}")
    typer.echo(
        f"vocabulary: retrieval {len(index.vocabulary)} terms; spelling "
        f"{len(spelling_df)} terms from {list(settings.query.spelling_fields)}"
    )
    typer.echo(
        f"parser build (word list, spelling index, brands): {build_ms:.0f} ms, "
        f"peak {peak_mb:.1f} MiB"
    )
    if parser.speller is not None:
        looked_up = sorted({t for q in queries for t in tokenize(q)})
        sizes = [len(parser.speller.candidates(t)) for t in looked_up]
        typer.echo(
            f"correction candidates per workload token: n={len(sizes)} "
            f"mean={statistics.fmean(sizes):.1f} max={max(sizes)}"
        )
        corrections = sorted(
            {
                f"{fix.original}->{fix.corrected} ({fix.reason.value})"
                for q in queries
                for fix in parser.parse(q).corrections
            }
        )
        typer.echo(f"corrections on workload ({len(corrections)}): {', '.join(corrections)}")
    if stages.brands:
        brands = sorted({f"{q!r}->{p.brand}" for q in queries if (p := parser.parse(q)).brand})
        notes = sorted({n for q in queries for n in parser.parse(q).brand_notes})
        typer.echo(f"brands recognised on workload ({len(brands)}): {', '.join(brands)}")
        typer.echo(f"brand notes ({len(notes)}): {' | '.join(notes) or '-'}")
    for label, samples in (
        ("parse", parse_ms),
        ("parse + catalog search, search part", catalog_ms),
        (f"store search (limit {limit})", store_ms),
    ):
        p50 = statistics.median(samples)
        p95 = statistics.quantiles(samples, n=100, method="inclusive")[94]
        typer.echo(f"{label}: n={len(samples)} p50={p50:.3f} ms p95={p95:.3f} ms")
    typer.echo(
        f"catalog zero-result over {len(queries)} dev query texts: plain BM25F {plain_zero}, "
        f"{stage.value} {stage_zero}"
    )


@app.command("evaluate-qu")
def evaluate_qu_command(
    snapshot_id: str,
    stage: Annotated[StageChoice, typer.Option(help="Query-understanding stage.")] = (
        StageChoice.S5
    ),
    expectations: Annotated[Path, typer.Option(help="Expectations file.")] = Path(
        "eval/query_understanding/synthetic-v1.toml"
    ),
) -> None:
    """Score query understanding (quantity, brand, spelling) against hand-written
    expectations for the snapshot's catalog. Development queries only."""
    settings = Settings()
    manifest, products, _ = _load_snapshot(settings, snapshot_id)
    document, sha256 = load_expectations(expectations)
    fingerprint = catalog_fingerprint(products)
    suites = [s for s in document.suite if s.catalog == manifest.source.name]
    if not suites or suites[0].catalog_sha256 != fingerprint:
        typer.echo(
            f"FAILED: no suite for catalog {manifest.source.name!r} with fingerprint {fingerprint}",
            err=True,
        )
        raise typer.Exit(1)
    index = BM25FIndex.build(products, settings.retrieval.bm25)
    stages = STAGES[stage.value]
    parser = build_parser(products, index.vocabulary, settings.query, stages)
    identity = json.dumps(
        {
            "bm25": bm25_config_id(settings.retrieval.bm25),
            **parser_identity(settings.query, stages),
        },
        sort_keys=True,
    )
    pipeline = f"qu-{stage.value}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
    workload = [q.text for q in load_judgments(settings.evaluation.judgments_file).queries]
    report = evaluate_suite(
        suites[0], parser.parse, workload, pipeline=pipeline, expectations_sha256=sha256
    )
    path = (
        settings.paths.artifacts_dir
        / "eval-qu"
        / snapshot_id
        / f"{pipeline}-{suites[0].name}-{sha256[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")
    typer.echo(format_suite_report(report))
    typer.echo(f"report: {path}")


@app.command("evaluate-tiers")
def evaluate_tiers_command(
    snapshot_id: str,
    stage: Annotated[StageChoice, typer.Option(help="Stage with ordering.")] = StageChoice.S6,
    judgments_path: Annotated[
        Path | None,
        typer.Option("--judgments", help="Judgment file (defaults to evaluation.judgments_file)."),
    ] = None,
) -> None:
    """EXACT-tier precision/recall against grade-3 labels, and exact/relaxed/candidate
    counts. Without labels for this catalog, counts only (query texts from the file)."""
    settings = Settings()
    manifest, products, in_stock = _load_snapshot(settings, snapshot_id)
    if not STAGES[stage.value].ordering:
        raise typer.BadParameter("tiers exist only with ordering (s6)", param_hint="--stage")
    judgments = load_judgments(judgments_path or settings.evaluation.judgments_file)
    labelled = (
        judgments.catalog == manifest.source.name
        and judgments.catalog_sha256 == catalog_fingerprint(products)
    )
    pipeline = UnderstandingPipeline(
        products,
        in_stock,
        settings.retrieval.bm25,
        settings.query,
        stage.value,
        STAGES[stage.value],
        settings.ranking,
    )
    report = evaluate_tiers(
        pipeline,
        [TierQuery(q.query_id, q.text, q if labelled else None) for q in judgments.queries],
        {product.product_id: product for product in products},
        snapshot_id=snapshot_id,
        judgments_sha256=judgments.sha256,
        candidate_depth=settings.ranking.candidate_depth,
        k=settings.evaluation.k,
    )
    suffix = judgments.sha256[:12] if labelled else "unlabelled"
    path = (
        settings.paths.artifacts_dir
        / "eval-tiers"
        / snapshot_id
        / f"{report.pipeline}-{suffix}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")
    typer.echo(format_tiers_report(report))
    typer.echo(f"report: {path}")
