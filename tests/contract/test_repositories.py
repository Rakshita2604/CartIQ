"""One behavioural contract, run against every storage implementation."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from cartiq.domain import Order, OrderLine, Product, Quantity, StockRecord, Store, Unit, User
from cartiq.storage import IntegrityError, Repositories
from cartiq.storage.memory import in_memory_repositories
from cartiq.storage.sqlite import connect, initialize_schema, sqlite_repositories

T = datetime(2026, 3, 1, 10, 30, tzinfo=UTC)

MILK = Product(
    product_id="p-milk",
    name="Amul Taaza Toned Milk 1 l",
    category_path=("Dairy", "Milk"),
    price=Decimal("54.00"),
    brand="Amul",
    quantity=Quantity(1, Unit.L),
    mrp=Decimal("56.00"),
    tags=("toned",),
    description="Toned milk.",
)
ONION = Product("p-onion", "Fresh Onion 1 kg", ("Vegetables",), Decimal("40.00"))
COLA = replace(MILK, product_id="p-cola", name="Cola 6 x 300 ml", pack_count=6, mrp=None)
STORE_A = Store("s-a", "Store A", "Bengaluru")
STORE_B = Store("s-b", "Store B", "Mumbai")


def _memory(_: Path) -> Iterator[Repositories]:
    yield in_memory_repositories()


def _sqlite(tmp_path: Path) -> Iterator[Repositories]:
    conn = connect(tmp_path / "test.sqlite")
    initialize_schema(conn)
    yield sqlite_repositories(conn)
    conn.close()


@pytest.fixture(params=[_memory, _sqlite], ids=["memory", "sqlite"])
def repos(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Repositories]:
    factory: Callable[[Path], Iterator[Repositories]] = request.param
    yield from factory(tmp_path)


@pytest.fixture
def seeded(repos: Repositories) -> Repositories:
    repos.catalog.add_many([ONION, MILK, COLA])
    repos.stores.add_many([STORE_B, STORE_A])
    repos.users.add_many([User("u-2", "s-b"), User("u-1", "s-a"), User("u-3")])
    return repos


class TestCatalog:
    def test_round_trips_every_field(self, seeded: Repositories) -> None:
        assert seeded.catalog.get("p-milk") == MILK
        assert seeded.catalog.get("p-cola") == COLA
        assert seeded.catalog.get("p-onion") == ONION

    def test_list_all_is_sorted_and_count_matches(self, seeded: Repositories) -> None:
        assert [p.product_id for p in seeded.catalog.list_all()] == ["p-cola", "p-milk", "p-onion"]
        assert seeded.catalog.count() == 3

    def test_get_many_omits_unknown_ids(self, seeded: Repositories) -> None:
        found = seeded.catalog.get_many(["p-milk", "missing", "p-milk"])

        assert found == {"p-milk": MILK}
        assert seeded.catalog.get("missing") is None

    def test_get_many_handles_large_id_lists(self, seeded: Repositories) -> None:
        ids = [f"x-{i}" for i in range(2_000)] + ["p-onion"]

        assert seeded.catalog.get_many(ids) == {"p-onion": ONION}

    def test_duplicate_insert_is_rejected_atomically(self, seeded: Repositories) -> None:
        new = replace(ONION, product_id="p-new")
        with pytest.raises(IntegrityError):
            seeded.catalog.add_many([new, MILK])

        assert seeded.catalog.get("p-new") is None


class TestStores:
    def test_round_trip_and_order(self, seeded: Repositories) -> None:
        assert seeded.stores.list_all() == [STORE_A, STORE_B]
        assert seeded.stores.get("s-b") == STORE_B
        assert seeded.stores.get("nope") is None


class TestStock:
    def test_distinguishes_not_carried_from_out_of_stock(self, seeded: Repositories) -> None:
        seeded.stock.upsert_many(
            [StockRecord("s-a", "p-milk", 5, T), StockRecord("s-a", "p-onion", 0, T)]
        )

        records = seeded.stock.get_many("s-a", ["p-milk", "p-onion", "p-cola"])

        assert set(records) == {"p-milk", "p-onion"}
        assert not records["p-onion"].in_stock
        assert seeded.stock.in_stock_ids("s-a") == {"p-milk"}
        assert seeded.stock.in_stock_ids("s-b") == frozenset()

    def test_upsert_replaces_existing_record(self, seeded: Repositories) -> None:
        later = datetime(2026, 3, 2, tzinfo=UTC)
        seeded.stock.upsert_many([StockRecord("s-a", "p-milk", 5, T)])
        seeded.stock.upsert_many([StockRecord("s-a", "p-milk", 0, later)])

        assert seeded.stock.list_all() == [StockRecord("s-a", "p-milk", 0, later)]
        assert seeded.stock.in_stock_ids("s-a") == frozenset()

    def test_list_all_sorted_by_store_then_product(self, seeded: Repositories) -> None:
        seeded.stock.upsert_many(
            [
                StockRecord("s-b", "p-milk", 1, T),
                StockRecord("s-a", "p-onion", 1, T),
                StockRecord("s-a", "p-cola", 1, T),
            ]
        )

        keys = [(r.store_id, r.product_id) for r in seeded.stock.list_all()]
        assert keys == [("s-a", "p-cola"), ("s-a", "p-onion"), ("s-b", "p-milk")]

    @pytest.mark.parametrize(
        "record",
        [StockRecord("s-x", "p-milk", 1, T), StockRecord("s-a", "p-x", 1, T)],
        ids=["unknown-store", "unknown-product"],
    )
    def test_rejects_dangling_references(self, seeded: Repositories, record: StockRecord) -> None:
        with pytest.raises(IntegrityError):
            seeded.stock.upsert_many([record])


class TestUsers:
    def test_round_trip(self, seeded: Repositories) -> None:
        assert seeded.users.get("u-3") == User("u-3")
        assert [u.user_id for u in seeded.users.list_all()] == ["u-1", "u-2", "u-3"]

    def test_rejects_unknown_home_store(self, seeded: Repositories) -> None:
        with pytest.raises(IntegrityError):
            seeded.users.add_many([User("u-9", "s-x")])


class TestOrders:
    def _order(self, order_id: str, user_id: str, at: datetime, *lines: OrderLine) -> Order:
        return Order(order_id, user_id, "s-a", at, lines)

    def test_round_trip_preserves_line_order(self, seeded: Repositories) -> None:
        order = self._order("o-1", "u-1", T, OrderLine("p-onion", 2), OrderLine("p-milk", 1))
        seeded.orders.add_many([order])

        assert seeded.orders.list_all() == [order]

    def test_for_user_is_chronological(self, seeded: Repositories) -> None:
        late = self._order("o-a", "u-1", datetime(2026, 3, 5, tzinfo=UTC), OrderLine("p-milk", 1))
        early = self._order("o-b", "u-1", T, OrderLine("p-milk", 1))
        other = self._order("o-c", "u-2", T, OrderLine("p-milk", 1))
        seeded.orders.add_many([late, early, other])

        assert seeded.orders.for_user("u-1") == [early, late]
        assert seeded.orders.for_user("u-9") == []
        assert [o.order_id for o in seeded.orders.list_all()] == ["o-a", "o-b", "o-c"]

    @pytest.mark.parametrize(
        "order",
        [
            Order("o-x", "u-x", "s-a", T, (OrderLine("p-milk", 1),)),
            Order("o-x", "u-1", "s-x", T, (OrderLine("p-milk", 1),)),
            Order("o-x", "u-1", "s-a", T, (OrderLine("p-x", 1),)),
            Order("o-x", "u-1", "s-a", T, (OrderLine("p-milk", 1), OrderLine("p-milk", 2))),
        ],
        ids=["unknown-user", "unknown-store", "unknown-product", "repeated-product"],
    )
    def test_rejects_invalid_orders(self, seeded: Repositories, order: Order) -> None:
        with pytest.raises(IntegrityError):
            seeded.orders.add_many([order])

        assert seeded.orders.list_all() == []
