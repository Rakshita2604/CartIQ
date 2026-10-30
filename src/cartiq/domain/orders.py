"""Users and order history, the raw input to popularity and personalization."""

from dataclasses import dataclass
from datetime import datetime

from cartiq.domain._validation import require_aware, require_id


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    home_store_id: str | None = None

    def __post_init__(self) -> None:
        require_id("User.user_id", self.user_id)


@dataclass(frozen=True, slots=True)
class OrderLine:
    product_id: str
    units: int

    def __post_init__(self) -> None:
        require_id("OrderLine.product_id", self.product_id)
        if self.units < 1:
            raise ValueError("OrderLine.units must be >= 1")


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    user_id: str
    store_id: str
    placed_at: datetime
    lines: tuple[OrderLine, ...]

    def __post_init__(self) -> None:
        require_id("Order.order_id", self.order_id)
        require_id("Order.user_id", self.user_id)
        require_id("Order.store_id", self.store_id)
        require_aware("Order.placed_at", self.placed_at)
        if not self.lines:
            raise ValueError("Order.lines must not be empty")
