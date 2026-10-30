from dataclasses import replace
from decimal import Decimal

import pytest

from cartiq.domain import Quantity, Unit
from cartiq.ingestion.normalize import normalize_catalog
from cartiq.ingestion.raw import RawCatalogRow

ROW = RawCatalogRow(
    source_id="x-1",
    name="Amul Taaza Toned Milk 500 ml",
    category="Bakery, Cakes & Dairy",
    sub_category="Dairy",
    product_type="Milk",
    brand="Amul",
    sale_price="28",
    market_price="30",
    description="Toned milk.",
)


def test_normalizes_a_valid_row() -> None:
    (product,) = normalize_catalog([ROW]).products

    assert product.product_id == "x-1"
    assert product.category_path == ("Bakery, Cakes & Dairy", "Dairy", "Milk")
    assert product.price == Decimal("28.00")
    assert product.mrp == Decimal("30.00")
    assert product.quantity == Quantity(500, Unit.ML)


def test_cleans_whitespace_and_unicode() -> None:
    row = replace(ROW, name="  Amul\u00a0Taaza   Milk ", brand=" Amul ", product_type=" ")

    (product,) = normalize_catalog([row]).products

    assert product.name == "Amul Taaza Milk"
    assert product.brand == "Amul"
    assert product.category_path == ("Bakery, Cakes & Dairy", "Dairy")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"source_id": " "}, "missing_id"),
        ({"name": ""}, "missing_name"),
        ({"category": "", "sub_category": "", "product_type": ""}, "missing_category"),
        ({"sale_price": "free"}, "invalid_price"),
        ({"sale_price": "-5"}, "invalid_price"),
        ({"sale_price": "NaN"}, "invalid_price"),
    ],
)
def test_rejects_invalid_rows_with_reason(overrides: dict[str, str], reason: str) -> None:
    catalog = normalize_catalog([replace(ROW, **overrides)])

    assert catalog.products == ()
    assert catalog.report.rejected == {reason: 1}


def test_drops_mrp_below_price_but_keeps_product() -> None:
    catalog = normalize_catalog([replace(ROW, market_price="20")])

    assert catalog.products[0].mrp is None
    assert catalog.report.mrp_dropped == 1


def test_keeps_first_of_duplicate_ids() -> None:
    catalog = normalize_catalog([ROW, replace(ROW, name="Something Else")])

    assert [p.name for p in catalog.products] == [ROW.name]
    assert catalog.report.rejected == {"duplicate_id": 1}


def test_drops_content_duplicates_case_insensitively() -> None:
    duplicate = replace(ROW, source_id="x-2", name=ROW.name.upper(), brand="AMUL")

    catalog = normalize_catalog([ROW, duplicate])

    assert [p.product_id for p in catalog.products] == ["x-1"]
    assert catalog.report.duplicates_dropped == 1


def test_output_is_sorted_by_product_id() -> None:
    rows = [replace(ROW, source_id=sid, name=f"Milk {sid}") for sid in ("c", "a", "b")]

    assert [p.product_id for p in normalize_catalog(rows).products] == ["a", "b", "c"]
