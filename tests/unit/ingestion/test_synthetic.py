from cartiq.domain import Quantity, Unit
from cartiq.ingestion.normalize import normalize_catalog
from cartiq.ingestion.synthetic import synthetic_catalog_rows
from cartiq.snapshots import rows_sha256


def test_is_deterministic() -> None:
    assert rows_sha256(synthetic_catalog_rows()) == rows_sha256(synthetic_catalog_rows())


def test_normalizes_cleanly_with_full_quantity_coverage() -> None:
    catalog = normalize_catalog(synthetic_catalog_rows())
    report = catalog.report

    assert report.rows_read == report.accepted == len(catalog.products) > 100
    assert not report.rejected
    assert report.duplicates_dropped == 0
    assert report.quantity_parsed == report.accepted


def test_covers_cases_later_phases_rely_on() -> None:
    products = normalize_catalog(synthetic_catalog_rows()).products
    by_name = {p.name: p for p in products}
    top_categories_by_brand: dict[str, set[str]] = {}
    for p in products:
        if p.brand:
            top_categories_by_brand.setdefault(p.brand, set()).add(p.category_path[0])

    assert any(p.brand is None for p in products), "loose produce without brand"
    assert any(p.pack_count > 1 for p in products), "multi-packs"
    assert len(top_categories_by_brand["Amul"]) > 1, "brand spanning categories"
    assert by_name["Maggi 2-Minute Masala Noodles 12 x 70 g"].quantity == Quantity(70, Unit.G)
    assert by_name["Maggi 2-Minute Masala Noodles 12 x 70 g"].pack_count == 12
    assert {p.category_path[0] for p in products} >= {
        "Bakery, Cakes & Dairy",
        "Foodgrains, Oil & Masala",
        "Fruits & Vegetables",
        "Beverages",
        "Snacks & Branded Foods",
        "Cleaning & Household",
        "Beauty & Hygiene",
    }
