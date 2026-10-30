import hashlib
import unicodedata
from dataclasses import replace
from decimal import Decimal

import pytest

from cartiq.catalog_fingerprint import catalog_fingerprint
from cartiq.domain import Product, Quantity, Unit

LATTE = Product(
    "P1",
    "Café Latte",
    ("Beverages", "Coffee"),
    Decimal("120.00"),
    brand="Nescafé",
    quantity=Quantity(0.5, Unit.L),
    tags=("b", "a"),
)
LEMON = Product("P0", "Lemon", ("Fruits",), Decimal("30.00"), quantity=Quantity(250.0, Unit.G))


def test_matches_the_documented_canonical_form() -> None:
    # Written by hand from the module docstring: sorted ids, sorted keys, compact JSON,
    # null for absent values, integral numbers as integers, tags sorted, UTF-8.
    canonical = (
        "cartiq-catalog-fingerprint/v1\n"
        '{"brand":null,"category_path":["Fruits"],"description":"","id":"P0",'
        '"name":"Lemon","pack_count":1,"quantity":{"unit":"g","value":"250"},"tags":[]}\n'
        '{"brand":"Nescafé","category_path":["Beverages","Coffee"],"description":"","id":"P1",'
        '"name":"Café Latte","pack_count":1,"quantity":{"unit":"l","value":"0.5"},'
        '"tags":["a","b"]}\n'
    )

    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert catalog_fingerprint([LATTE, LEMON]) == expected


def test_independent_of_input_order() -> None:
    assert catalog_fingerprint([LATTE, LEMON]) == catalog_fingerprint([LEMON, LATTE])


def test_independent_of_unicode_normalization_form() -> None:
    decomposed = replace(LATTE, name=unicodedata.normalize("NFD", LATTE.name), brand="Nescafé")

    assert catalog_fingerprint([decomposed]) == catalog_fingerprint([LATTE])


def test_independent_of_tag_order() -> None:
    assert catalog_fingerprint([replace(LATTE, tags=("a", "b"))]) == catalog_fingerprint([LATTE])


def test_ignores_price_and_mrp() -> None:
    repriced = replace(LATTE, price=Decimal("99.00"), mrp=Decimal("150.00"))

    assert catalog_fingerprint([repriced]) == catalog_fingerprint([LATTE])


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Cafe Mocha"},
        {"brand": None},
        {"category_path": ("Beverages", "Tea")},
        {"quantity": Quantity(1, Unit.L)},
        {"quantity": None},
        {"pack_count": 6},
        {"tags": ("a",)},
        {"description": "Strong."},
    ],
)
def test_relevance_fields_change_the_fingerprint(change: dict[str, object]) -> None:
    assert catalog_fingerprint([replace(LATTE, **change)]) != catalog_fingerprint([LATTE])  # type: ignore[arg-type]


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        catalog_fingerprint([LATTE, replace(LATTE, name="Other")])
