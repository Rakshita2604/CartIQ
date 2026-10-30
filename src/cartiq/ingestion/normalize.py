"""Validate and normalize raw catalog rows into Products, with an auditable report."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from cartiq.domain import Product
from cartiq.ingestion.raw import RawCatalogRow
from cartiq.parsing.quantity import parse_quantity
from cartiq.parsing.text import clean_text

_CENTS = Decimal("0.01")


@dataclass
class IngestionReport:
    rows_read: int = 0
    accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    duplicates_dropped: int = 0
    mrp_dropped: int = 0
    quantity_parsed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_read": self.rows_read,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
            "duplicates_dropped": self.duplicates_dropped,
            "mrp_dropped": self.mrp_dropped,
            "quantity_parsed": self.quantity_parsed,
        }


@dataclass(frozen=True)
class NormalizedCatalog:
    products: tuple[Product, ...]
    report: IngestionReport


class _RowRejectedError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def normalize_catalog(rows: Iterable[RawCatalogRow]) -> NormalizedCatalog:
    """Keep the first occurrence of each product; output is sorted by product_id."""
    report = IngestionReport()
    seen_ids: set[str] = set()
    seen_keys: set[tuple[object, ...]] = set()
    products: list[Product] = []

    for row in rows:
        report.rows_read += 1
        try:
            product, mrp_dropped = _to_product(row)
        except _RowRejectedError as rejected:
            report.rejected[rejected.reason] += 1
            continue
        if product.product_id in seen_ids:
            report.rejected["duplicate_id"] += 1
            continue
        key = (
            product.name.casefold(),
            (product.brand or "").casefold(),
            product.category_path,
            product.price,
        )
        if key in seen_keys:
            report.duplicates_dropped += 1
            continue
        seen_ids.add(product.product_id)
        seen_keys.add(key)
        products.append(product)
        report.accepted += 1
        report.mrp_dropped += mrp_dropped
        report.quantity_parsed += product.quantity is not None

    products.sort(key=lambda p: p.product_id)
    return NormalizedCatalog(tuple(products), report)


def _to_product(row: RawCatalogRow) -> tuple[Product, bool]:
    product_id = clean_text(row.source_id)
    name = clean_text(row.name)
    if not product_id:
        raise _RowRejectedError("missing_id")
    if not name:
        raise _RowRejectedError("missing_name")

    levels = (row.category, row.sub_category, row.product_type)
    category_path = tuple(cleaned for cleaned in map(clean_text, levels) if cleaned)
    if not category_path:
        raise _RowRejectedError("missing_category")

    price = _parse_money(row.sale_price)
    if price is None or price <= 0:
        raise _RowRejectedError("invalid_price")
    mrp = _parse_money(row.market_price)
    mrp_dropped = mrp is not None and mrp < price
    if mrp_dropped:
        mrp = None

    parsed = parse_quantity(name)
    try:
        product = Product(
            product_id=product_id,
            name=name,
            category_path=category_path,
            price=price,
            brand=clean_text(row.brand) or None,
            quantity=parsed.quantity if parsed else None,
            pack_count=parsed.pack_count if parsed else 1,
            mrp=mrp,
            description=clean_text(row.description),
        )
    except ValueError as error:
        raise _RowRejectedError("invalid_record") from error
    return product, mrp_dropped


def _parse_money(raw: str) -> Decimal | None:
    text = clean_text(raw).replace(",", "")
    if not text:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value.quantize(_CENTS) if value.is_finite() else None
