"""Storage contracts. Business logic depends on these, never on a concrete database.

Shared contract for every implementation (enforced by tests/contract):
- ``list_*`` results are sorted by primary key, so iteration order is deterministic.
- ``get_many`` silently omits unknown ids.
- Inserting a duplicate key or a dangling reference raises ``IntegrityError``.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from cartiq.domain import Order, Product, StockRecord, Store, User


class IntegrityError(Exception):
    """A write violated a uniqueness or referential constraint."""


class CatalogRepository(Protocol):
    def add_many(self, products: Iterable[Product]) -> None: ...
    def get(self, product_id: str) -> Product | None: ...
    def get_many(self, product_ids: Iterable[str]) -> dict[str, Product]: ...
    def list_all(self) -> list[Product]: ...
    def count(self) -> int: ...


class StoreRepository(Protocol):
    def add_many(self, stores: Iterable[Store]) -> None: ...
    def get(self, store_id: str) -> Store | None: ...
    def list_all(self) -> list[Store]: ...


class StockRepository(Protocol):
    def upsert_many(self, records: Iterable[StockRecord]) -> None: ...

    def get_many(self, store_id: str, product_ids: Iterable[str]) -> dict[str, StockRecord]:
        """Stock records for carried products; products the store doesn't carry are omitted."""
        ...

    def in_stock_ids(self, store_id: str) -> frozenset[str]:
        """Products with available_qty > 0 at the store (used for store-aware retrieval)."""
        ...

    def list_all(self) -> list[StockRecord]: ...


class UserRepository(Protocol):
    def add_many(self, users: Iterable[User]) -> None: ...
    def get(self, user_id: str) -> User | None: ...
    def list_all(self) -> list[User]: ...


class OrderRepository(Protocol):
    def add_many(self, orders: Iterable[Order]) -> None: ...

    def for_user(self, user_id: str) -> list[Order]:
        """The user's orders, oldest first (ties broken by order_id)."""
        ...

    def list_all(self) -> list[Order]: ...


@dataclass(frozen=True, slots=True)
class Repositories:
    catalog: CatalogRepository
    stores: StoreRepository
    stock: StockRepository
    users: UserRepository
    orders: OrderRepository
