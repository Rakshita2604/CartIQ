from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawCatalogRow:
    """Source-agnostic catalog record before validation. All fields are untrusted strings."""

    source_id: str
    name: str
    category: str
    sub_category: str
    product_type: str
    brand: str
    sale_price: str
    market_price: str
    description: str
