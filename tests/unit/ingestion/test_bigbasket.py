from pathlib import Path

import pytest

from cartiq.domain import Quantity, Unit
from cartiq.ingestion.bigbasket import SchemaError, read_bigbasket_csv
from cartiq.ingestion.normalize import normalize_catalog

SAMPLE = Path(__file__).parents[2] / "fixtures" / "bigbasket_sample.csv"


def test_reads_rows_with_prefixed_ids() -> None:
    rows = list(read_bigbasket_csv(SAMPLE))

    assert len(rows) == 8
    assert rows[0].source_id == "bb-1"
    assert rows[1].sub_category == "Noodle, Pasta, Vermicelli"


def test_sample_normalizes_with_expected_report() -> None:
    catalog = normalize_catalog(read_bigbasket_csv(SAMPLE))

    assert catalog.report.to_dict() == {
        "rows_read": 8,
        "accepted": 4,
        "rejected": {"invalid_price": 2, "missing_name": 1},
        "duplicates_dropped": 1,
        "mrp_dropped": 1,
        "quantity_parsed": 3,
    }
    by_id = {p.product_id: p for p in catalog.products}
    noodles = by_id["bb-2"]
    assert noodles.name == "Masala Noodles 12 x 70 g"
    assert noodles.quantity == Quantity(70, Unit.G)
    assert noodles.pack_count == 12
    assert by_id["bb-1"].brand == "Amul"
    assert by_id["bb-3"].brand is None


def test_rejects_unexpected_header(tmp_path: Path) -> None:
    path = tmp_path / "wrong.csv"
    path.write_text("id,title,price\n1,Milk,54\n")

    with pytest.raises(SchemaError, match="missing columns"):
        list(read_bigbasket_csv(path))
