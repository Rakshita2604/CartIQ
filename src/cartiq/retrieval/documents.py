"""Turn a product into per-field token lists for lexical indexing.

Category levels map to fields by position: the first level is CATEGORY (a broad shelf such
as "Foodgrains, Oil & Masala"), the last is PRODUCT_TYPE ("Mustard Oil"), and any levels in
between form SUB_CATEGORY. A one-level path is only CATEGORY.
"""

from enum import StrEnum

from cartiq.domain import Product
from cartiq.parsing.text import tokenize


class Field(StrEnum):
    NAME = "name"
    PRODUCT_TYPE = "product_type"
    BRAND = "brand"
    SUB_CATEGORY = "sub_category"
    DESCRIPTION = "description"
    CATEGORY = "category"


def product_fields(product: Product) -> dict[Field, list[str]]:
    brand = tokenize(product.brand) if product.brand else []
    path = product.category_path
    return {
        Field.NAME: strip_brand_prefix(tokenize(product.name), brand),
        Field.PRODUCT_TYPE: tokenize(path[-1]) if len(path) > 1 else [],
        Field.BRAND: brand,
        Field.SUB_CATEGORY: [token for level in path[1:-1] for token in tokenize(level)],
        Field.DESCRIPTION: tokenize(product.description),
        Field.CATEGORY: tokenize(path[0]),
    }


def strip_brand_prefix(name: list[str], brand: list[str]) -> list[str]:
    """Drop the brand's tokens from the start of the name, so the brand is counted once (in
    its own field). Only a full leading match is removed, and never the whole name."""
    if brand and len(name) > len(brand) and name[: len(brand)] == brand:
        return name[len(brand) :]
    return name
