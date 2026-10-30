"""Catalog entities: products and their pack quantities.

Products carry catalog-wide facts only; per-store availability lives in inventory.
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from cartiq.domain._validation import require_id, require_positive


class Dimension(StrEnum):
    MASS = "mass"
    VOLUME = "volume"
    COUNT = "count"


class Unit(StrEnum):
    G = "g"
    KG = "kg"
    ML = "ml"
    L = "l"
    PCS = "pcs"

    @property
    def dimension(self) -> Dimension:
        return _UNIT_DIMENSION[self]


_UNIT_DIMENSION = {
    Unit.G: Dimension.MASS,
    Unit.KG: Dimension.MASS,
    Unit.ML: Dimension.VOLUME,
    Unit.L: Dimension.VOLUME,
    Unit.PCS: Dimension.COUNT,
}

# Conversion of each unit into the base unit of its dimension (g, ml, pcs).
_TO_BASE: dict[Unit, tuple[Unit, int]] = {
    Unit.G: (Unit.G, 1),
    Unit.KG: (Unit.G, 1000),
    Unit.ML: (Unit.ML, 1),
    Unit.L: (Unit.ML, 1000),
    Unit.PCS: (Unit.PCS, 1),
}


@dataclass(frozen=True, slots=True)
class Quantity:
    """Amount of a single unit of a product, e.g. 500 g or 1 l."""

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        require_positive("Quantity.value", self.value)

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def to_base(self) -> "Quantity":
        base_unit, factor = _TO_BASE[self.unit]
        return Quantity(self.value * factor, base_unit)

    def is_comparable(self, other: "Quantity") -> bool:
        return self.dimension is other.dimension

    def same_amount(self, other: "Quantity") -> bool:
        """True when both quantities describe the same amount, e.g. 1 kg and 1000 g."""
        if not self.is_comparable(other):
            return False
        return math.isclose(self.to_base().value, other.to_base().value, rel_tol=1e-9)


@dataclass(frozen=True, slots=True)
class Product:
    product_id: str
    name: str
    category_path: tuple[str, ...]
    price: Decimal
    brand: str | None = None
    quantity: Quantity | None = None
    pack_count: int = 1
    mrp: Decimal | None = None
    tags: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        require_id("Product.product_id", self.product_id)
        if not self.name.strip():
            raise ValueError("Product.name must not be blank")
        if not self.category_path or any(not c.strip() for c in self.category_path):
            raise ValueError("Product.category_path must be non-empty with no blank levels")
        if self.brand is not None and not self.brand.strip():
            raise ValueError("Product.brand must be None or non-blank")
        if self.price <= 0:
            raise ValueError("Product.price must be > 0")
        if self.mrp is not None and self.mrp < self.price:
            raise ValueError("Product.mrp must be >= price")
        if self.pack_count < 1:
            raise ValueError("Product.pack_count must be >= 1")

    @property
    def total_quantity(self) -> Quantity | None:
        """Total amount across the pack, e.g. 6 x 200 ml -> 1200 ml."""
        if self.quantity is None:
            return None
        return Quantity(self.quantity.value * self.pack_count, self.quantity.unit)
