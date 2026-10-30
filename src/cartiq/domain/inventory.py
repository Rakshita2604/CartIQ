"""Store and stock entities. Availability is never mixed into relevance scoring."""

from dataclasses import dataclass
from datetime import datetime

from cartiq.domain._validation import require_aware, require_id


@dataclass(frozen=True, slots=True)
class Store:
    store_id: str
    name: str
    city: str

    def __post_init__(self) -> None:
        require_id("Store.store_id", self.store_id)


@dataclass(frozen=True, slots=True)
class StockRecord:
    store_id: str
    product_id: str
    available_qty: int
    updated_at: datetime

    def __post_init__(self) -> None:
        require_id("StockRecord.store_id", self.store_id)
        require_id("StockRecord.product_id", self.product_id)
        if self.available_qty < 0:
            raise ValueError("StockRecord.available_qty must be >= 0")
        require_aware("StockRecord.updated_at", self.updated_at)

    @property
    def in_stock(self) -> bool:
        return self.available_qty > 0
