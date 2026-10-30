from decimal import Decimal

from cartiq.domain import Product
from cartiq.retrieval.documents import Field, product_fields, strip_brand_prefix


def make(name: str, brand: str | None, path: tuple[str, ...], description: str = "") -> Product:
    return Product("P1", name, path, Decimal(10), brand=brand, description=description)


def test_maps_category_levels_by_position() -> None:
    fields = product_fields(
        make(
            "Fortune Kachi Ghani Mustard Oil 1 l",
            "Fortune",
            ("Foodgrains, Oil & Masala", "Edible Oils & Ghee", "Mustard Oil"),
            "Cold pressed.",
        )
    )

    assert fields == {
        Field.NAME: ["kachi", "ghani", "mustard", "oil", "1", "l"],
        Field.PRODUCT_TYPE: ["mustard", "oil"],
        Field.BRAND: ["fortune"],
        Field.SUB_CATEGORY: ["edible", "oils", "ghee"],
        Field.DESCRIPTION: ["cold", "pressed"],
        Field.CATEGORY: ["foodgrains", "oil", "masala"],
    }


def test_short_category_paths() -> None:
    two = product_fields(make("Lemon", None, ("Fruits", "Citrus")))
    one = product_fields(make("Lemon", None, ("Fruits",)))

    assert (two[Field.CATEGORY], two[Field.SUB_CATEGORY], two[Field.PRODUCT_TYPE]) == (
        ["fruits"],
        [],
        ["citrus"],
    )
    assert (one[Field.CATEGORY], one[Field.PRODUCT_TYPE]) == (["fruits"], [])


def test_missing_brand_leaves_name_intact() -> None:
    fields = product_fields(make("Fresh Onion 1 kg", None, ("Veg", "Onion")))

    assert fields[Field.BRAND] == []
    assert fields[Field.NAME] == ["fresh", "onion", "1", "kg"]


class TestStripBrandPrefix:
    def test_strips_a_full_leading_match(self) -> None:
        assert strip_brand_prefix(["mother", "dairy", "paneer"], ["mother", "dairy"]) == ["paneer"]

    def test_keeps_a_partial_or_non_leading_match(self) -> None:
        assert strip_brand_prefix(["mother", "paneer"], ["mother", "dairy"]) == ["mother", "paneer"]
        assert strip_brand_prefix(["real", "mixed", "juice"], ["mixed"]) == [
            "real",
            "mixed",
            "juice",
        ]

    def test_never_empties_the_name(self) -> None:
        assert strip_brand_prefix(["britannia"], ["britannia"]) == ["britannia"]

    def test_no_brand(self) -> None:
        assert strip_brand_prefix(["fresh", "onion"], []) == ["fresh", "onion"]
