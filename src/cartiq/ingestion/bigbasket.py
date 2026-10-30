"""Adapter for the Kaggle "BigBasket Entire Product List (~28K datapoints)" CSV.

Source: kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints
License: CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). The file is not
redistributed with this repository; download it manually into data/raw/.
"""

import csv
from collections.abc import Iterator
from pathlib import Path

from cartiq.ingestion.raw import RawCatalogRow

EXPECTED_COLUMNS = (
    "index",
    "product",
    "category",
    "sub_category",
    "brand",
    "sale_price",
    "market_price",
    "type",
    "rating",
    "description",
)


class SchemaError(ValueError):
    pass


def read_bigbasket_csv(path: Path) -> Iterator[RawCatalogRow]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(EXPECTED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise SchemaError(
                f"{path} is missing columns {sorted(missing)}; "
                f"found {reader.fieldnames}, expected {list(EXPECTED_COLUMNS)}"
            )
        for record in reader:
            yield RawCatalogRow(
                source_id=f"bb-{record['index'].strip()}",
                name=record["product"] or "",
                category=record["category"] or "",
                sub_category=record["sub_category"] or "",
                product_type=record["type"] or "",
                brand=record["brand"] or "",
                sale_price=record["sale_price"] or "",
                market_price=record["market_price"] or "",
                description=record["description"] or "",
            )
