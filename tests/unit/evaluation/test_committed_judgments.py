"""Guards on the judgment files committed under eval/judgments."""

from pathlib import Path

from cartiq.catalog_fingerprint import catalog_fingerprint
from cartiq.domain import Coverage, Grade, Product, QuerySegment
from cartiq.evaluation import check_against_catalog, load_judgments
from cartiq.ingestion.synthetic import SYNTHETIC_CATALOG_VERSION

SYNTHETIC_JUDGMENTS = Path(__file__).parents[3] / "eval" / "judgments" / "synthetic-v1.toml"


def test_synthetic_judgments_match_the_synthetic_catalog(
    synthetic_products: tuple[Product, ...],
) -> None:
    check_against_catalog(
        load_judgments(SYNTHETIC_JUDGMENTS),
        SYNTHETIC_CATALOG_VERSION,
        catalog_fingerprint(synthetic_products),
        {product.product_id for product in synthetic_products},
    )


def test_every_segment_is_covered() -> None:
    queries = load_judgments(SYNTHETIC_JUDGMENTS).queries

    for segment in QuerySegment:
        assert sum(query.segment is segment for query in queries) >= 5, segment


def test_every_grade_one_or_two_has_a_reason() -> None:
    for query in load_judgments(SYNTHETIC_JUDGMENTS).queries:
        for product_id, grade in query.grades.items():
            if grade in (Grade.RELATED, Grade.RELEVANT):
                assert product_id in query.reasons, (query.query_id, product_id)


def test_synthetic_queries_are_exhaustive() -> None:
    # The 140-product catalog was read in full for every query.
    queries = load_judgments(SYNTHETIC_JUDGMENTS).queries

    assert all(query.coverage is Coverage.EXHAUSTIVE for query in queries)


def test_quantity_queries_only_grade_exact_products_with_known_sizes(
    synthetic_products: tuple[Product, ...],
) -> None:
    sizes = {product.product_id: product.quantity for product in synthetic_products}

    for query in load_judgments(SYNTHETIC_JUDGMENTS).queries:
        if query.segment is QuerySegment.QUANTITY:
            exact = query.products_at_least(Grade.EXACT)
            assert all(sizes[pid] is not None for pid in exact), query.query_id
