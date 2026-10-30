"""Dict-backed repositories for tests and small fixtures. Same contract as SQLite."""

from collections.abc import Iterable, Mapping

from cartiq.domain import Order, Product, StockRecord, Store, User
from cartiq.storage.interfaces import IntegrityError, Repositories


def _insert_unique[T](target: dict[str, T], items: Iterable[tuple[str, T]], kind: str) -> None:
    staged: dict[str, T] = {}
    for key, item in items:
        if key in target or key in staged:
            raise IntegrityError(f"duplicate {kind} id: {key}")
        staged[key] = item
    target.update(staged)


def _require(known: Mapping[str, object], key: str | None, kind: str) -> None:
    if key is not None and key not in known:
        raise IntegrityError(f"unknown {kind} id: {key}")


class InMemoryCatalogRepository:
    def __init__(self) -> None:
        self.items: dict[str, Product] = {}

    def add_many(self, products: Iterable[Product]) -> None:
        _insert_unique(self.items, ((p.product_id, p) for p in products), "product")

    def get(self, product_id: str) -> Product | None:
        return self.items.get(product_id)

    def get_many(self, product_ids: Iterable[str]) -> dict[str, Product]:
        return {pid: self.items[pid] for pid in product_ids if pid in self.items}

    def list_all(self) -> list[Product]:
        return [self.items[pid] for pid in sorted(self.items)]

    def count(self) -> int:
        return len(self.items)


class InMemoryStoreRepository:
    def __init__(self) -> None:
        self.items: dict[str, Store] = {}

    def add_many(self, stores: Iterable[Store]) -> None:
        _insert_unique(self.items, ((s.store_id, s) for s in stores), "store")

    def get(self, store_id: str) -> Store | None:
        return self.items.get(store_id)

    def list_all(self) -> list[Store]:
        return [self.items[sid] for sid in sorted(self.items)]


class InMemoryStockRepository:
    def __init__(self, catalog: InMemoryCatalogRepository, stores: InMemoryStoreRepository):
        self._catalog = catalog
        self._stores = stores
        self._by_store: dict[str, dict[str, StockRecord]] = {}

    def upsert_many(self, records: Iterable[StockRecord]) -> None:
        staged = list(records)
        for record in staged:
            _require(self._stores.items, record.store_id, "store")
            _require(self._catalog.items, record.product_id, "product")
        for record in staged:
            self._by_store.setdefault(record.store_id, {})[record.product_id] = record

    def get_many(self, store_id: str, product_ids: Iterable[str]) -> dict[str, StockRecord]:
        carried = self._by_store.get(store_id, {})
        return {pid: carried[pid] for pid in product_ids if pid in carried}

    def in_stock_ids(self, store_id: str) -> frozenset[str]:
        carried = self._by_store.get(store_id, {})
        return frozenset(pid for pid, record in carried.items() if record.in_stock)

    def list_all(self) -> list[StockRecord]:
        return [
            self._by_store[sid][pid]
            for sid in sorted(self._by_store)
            for pid in sorted(self._by_store[sid])
        ]


class InMemoryUserRepository:
    def __init__(self, stores: InMemoryStoreRepository) -> None:
        self._stores = stores
        self.items: dict[str, User] = {}

    def add_many(self, users: Iterable[User]) -> None:
        staged = list(users)
        for user in staged:
            _require(self._stores.items, user.home_store_id, "store")
        _insert_unique(self.items, ((u.user_id, u) for u in staged), "user")

    def get(self, user_id: str) -> User | None:
        return self.items.get(user_id)

    def list_all(self) -> list[User]:
        return [self.items[uid] for uid in sorted(self.items)]


class InMemoryOrderRepository:
    def __init__(
        self,
        catalog: InMemoryCatalogRepository,
        stores: InMemoryStoreRepository,
        users: InMemoryUserRepository,
    ) -> None:
        self._catalog = catalog
        self._stores = stores
        self._users = users
        self._items: dict[str, Order] = {}

    def add_many(self, orders: Iterable[Order]) -> None:
        staged = list(orders)
        for order in staged:
            _require(self._users.items, order.user_id, "user")
            _require(self._stores.items, order.store_id, "store")
            for line in order.lines:
                _require(self._catalog.items, line.product_id, "product")
            if len({line.product_id for line in order.lines}) != len(order.lines):
                raise IntegrityError(f"duplicate product in order {order.order_id}")
        _insert_unique(self._items, ((o.order_id, o) for o in staged), "order")

    def for_user(self, user_id: str) -> list[Order]:
        orders = [o for o in self._items.values() if o.user_id == user_id]
        return sorted(orders, key=lambda o: (o.placed_at, o.order_id))

    def list_all(self) -> list[Order]:
        return [self._items[oid] for oid in sorted(self._items)]


def in_memory_repositories() -> Repositories:
    catalog = InMemoryCatalogRepository()
    stores = InMemoryStoreRepository()
    users = InMemoryUserRepository(stores)
    return Repositories(
        catalog=catalog,
        stores=stores,
        stock=InMemoryStockRepository(catalog, stores),
        users=users,
        orders=InMemoryOrderRepository(catalog, stores, users),
    )
