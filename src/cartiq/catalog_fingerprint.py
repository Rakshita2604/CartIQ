"""A stable fingerprint of the catalog fields that bear on relevance.

Judgment files record this fingerprint so labels are only ever applied to the catalog they
were made against. Price and MRP are excluded on purpose: a price change does not change
what a product is, so it must not invalidate labels.

Canonical form (version ``v1``), so the same catalog always gives the same hash:
- products sorted by NFC-normalized product id (Unicode code point order);
- one compact JSON object per product, keys sorted, UTF-8, newline-terminated;
- every string NFC-normalized; absent values (no brand, unknown size) are JSON ``null``;
- numbers written as integers when integral ("500"), else Python's shortest round-trip
  repr ("0.5"); tags sorted, category path kept in hierarchy order.
"""

import hashlib
import json
import unicodedata
from collections.abc import Iterable

from cartiq.domain import Product

FINGERPRINT_VERSION = "v1"


def catalog_fingerprint(products: Iterable[Product]) -> str:
    keyed = sorted(((_text(p.product_id), _canonical(p)) for p in products), key=lambda r: r[0])
    ids = [product_id for product_id, _ in keyed]
    if len(set(ids)) != len(ids):
        raise ValueError("catalog contains duplicate product ids")
    records = [record for _, record in keyed]

    digest = hashlib.sha256(f"cartiq-catalog-fingerprint/{FINGERPRINT_VERSION}\n".encode())
    for record in records:
        line = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _canonical(product: Product) -> dict[str, object]:
    quantity = product.quantity
    return {
        "id": _text(product.product_id),
        "name": _text(product.name),
        "brand": None if product.brand is None else _text(product.brand),
        "category_path": [_text(level) for level in product.category_path],
        "quantity": None
        if quantity is None
        else {"value": _number(quantity.value), "unit": quantity.unit.value},
        "pack_count": product.pack_count,
        "tags": sorted(_text(tag) for tag in product.tags),
        "description": _text(product.description),
    }


def _text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))
