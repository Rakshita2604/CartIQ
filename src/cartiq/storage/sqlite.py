"""SQLite-backed repositories. All SQL lives in this module."""

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from itertools import batched
from pathlib import Path

from cartiq.domain import Order, OrderLine, Product, Quantity, StockRecord, Store, Unit, User
from cartiq.storage.interfaces import IntegrityError, Repositories

SCHEMA_VERSION = 1
_IN_CLAUSE_BATCH = 500

_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE products (
    product_id     TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    brand          TEXT,
    category_path  TEXT NOT NULL,  -- JSON array
    price          TEXT NOT NULL,  -- decimal string
    mrp            TEXT,
    quantity_value REAL,
    quantity_unit  TEXT,
    pack_count     INTEGER NOT NULL CHECK (pack_count >= 1),
    tags           TEXT NOT NULL,  -- JSON array
    description    TEXT NOT NULL,
    CHECK ((quantity_value IS NULL) = (quantity_unit IS NULL))
);
CREATE TABLE stores (
    store_id TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    city     TEXT NOT NULL
);
CREATE TABLE stock (
    store_id      TEXT NOT NULL REFERENCES stores (store_id),
    product_id    TEXT NOT NULL REFERENCES products (product_id),
    available_qty INTEGER NOT NULL CHECK (available_qty >= 0),
    updated_at    TEXT NOT NULL,  -- ISO-8601 UTC
    PRIMARY KEY (store_id, product_id)
);
CREATE INDEX stock_in_stock ON stock (store_id, product_id) WHERE available_qty > 0;
CREATE TABLE users (
    user_id       TEXT PRIMARY KEY,
    home_store_id TEXT REFERENCES stores (store_id)
);
CREATE TABLE orders (
    order_id  TEXT PRIMARY KEY,
    user_id   TEXT NOT NULL REFERENCES users (user_id),
    store_id  TEXT NOT NULL REFERENCES stores (store_id),
    placed_at TEXT NOT NULL
);
CREATE INDEX orders_by_user ON orders (user_id, placed_at, order_id);
CREATE TABLE order_lines (
    order_id   TEXT NOT NULL REFERENCES orders (order_id),
    line_no    INTEGER NOT NULL,
    product_id TEXT NOT NULL REFERENCES products (product_id),
    units      INTEGER NOT NULL CHECK (units >= 1),
    PRIMARY KEY (order_id, line_no),
    UNIQUE (order_id, product_id)
);
"""


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode={'ro' if read_only else 'rwc'}"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,)
        )


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0])


def sqlite_repositories(conn: sqlite3.Connection) -> Repositories:
    """Repositories sharing one connection. The caller owns and closes the connection."""
    return Repositories(
        catalog=SqliteCatalogRepository(conn),
        stores=SqliteStoreRepository(conn),
        stock=SqliteStockRepository(conn),
        users=SqliteUserRepository(conn),
        orders=SqliteOrderRepository(conn),
    )


@contextmanager
def _write(conn: sqlite3.Connection) -> Iterator[None]:
    try:
        with conn:
            yield
    except sqlite3.IntegrityError as error:
        raise IntegrityError(str(error)) from error


def _select_in[T](
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[object],
    ids: Iterable[str],
    decode: Callable[[sqlite3.Row], T],
) -> list[T]:
    """Run ``sql`` (containing one ``{in}`` placeholder) over ``ids`` in bounded batches."""
    results: list[T] = []
    for chunk in batched(dict.fromkeys(ids), _IN_CLAUSE_BATCH):
        query = sql.format(**{"in": ",".join("?" * len(chunk))})
        results.extend(decode(row) for row in conn.execute(query, (*params, *chunk)))
    return results


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class SqliteCatalogRepository:
    _COLUMNS = (
        "product_id, name, brand, category_path, price, mrp, quantity_value, quantity_unit, "
        "pack_count, tags, description"
    )

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_many(self, products: Iterable[Product]) -> None:
        rows = [
            (
                p.product_id,
                p.name,
                p.brand,
                json.dumps(p.category_path),
                str(p.price),
                None if p.mrp is None else str(p.mrp),
                None if p.quantity is None else p.quantity.value,
                None if p.quantity is None else p.quantity.unit.value,
                p.pack_count,
                json.dumps(p.tags),
                p.description,
            )
            for p in products
        ]
        with _write(self._conn):
            self._conn.executemany(
                f"INSERT INTO products ({self._COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
            )

    def get(self, product_id: str) -> Product | None:
        return self.get_many([product_id]).get(product_id)

    def get_many(self, product_ids: Iterable[str]) -> dict[str, Product]:
        sql = f"SELECT {self._COLUMNS} FROM products WHERE product_id IN ({{in}})"
        products = _select_in(self._conn, sql, (), product_ids, self._decode)
        return {p.product_id: p for p in products}

    def list_all(self) -> list[Product]:
        sql = f"SELECT {self._COLUMNS} FROM products ORDER BY product_id"
        return [self._decode(row) for row in self._conn.execute(sql)]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    @staticmethod
    def _decode(row: sqlite3.Row) -> Product:
        (pid, name, brand, category, price, mrp, q_value, q_unit, pack, tags, desc) = row
        return Product(
            product_id=pid,
            name=name,
            brand=brand,
            category_path=tuple(json.loads(category)),
            price=Decimal(price),
            mrp=None if mrp is None else Decimal(mrp),
            quantity=None if q_value is None else Quantity(q_value, Unit(q_unit)),
            pack_count=pack,
            tags=tuple(json.loads(tags)),
            description=desc,
        )


class SqliteStoreRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_many(self, stores: Iterable[Store]) -> None:
        rows = [(s.store_id, s.name, s.city) for s in stores]
        with _write(self._conn):
            self._conn.executemany("INSERT INTO stores VALUES (?,?,?)", rows)

    def get(self, store_id: str) -> Store | None:
        row = self._conn.execute(
            "SELECT store_id, name, city FROM stores WHERE store_id = ?", (store_id,)
        ).fetchone()
        return None if row is None else Store(*row)

    def list_all(self) -> list[Store]:
        rows = self._conn.execute("SELECT store_id, name, city FROM stores ORDER BY store_id")
        return [Store(*row) for row in rows]


class SqliteStockRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_many(self, records: Iterable[StockRecord]) -> None:
        rows = [(r.store_id, r.product_id, r.available_qty, _iso(r.updated_at)) for r in records]
        with _write(self._conn):
            self._conn.executemany(
                "INSERT INTO stock VALUES (?,?,?,?) ON CONFLICT (store_id, product_id) DO UPDATE "
                "SET available_qty = excluded.available_qty, updated_at = excluded.updated_at",
                rows,
            )

    def get_many(self, store_id: str, product_ids: Iterable[str]) -> dict[str, StockRecord]:
        sql = (
            "SELECT store_id, product_id, available_qty, updated_at FROM stock "
            "WHERE store_id = ? AND product_id IN ({in})"
        )
        records = _select_in(self._conn, sql, (store_id,), product_ids, self._decode)
        return {r.product_id: r for r in records}

    def in_stock_ids(self, store_id: str) -> frozenset[str]:
        rows = self._conn.execute(
            "SELECT product_id FROM stock WHERE store_id = ? AND available_qty > 0", (store_id,)
        )
        return frozenset(row[0] for row in rows)

    def list_all(self) -> list[StockRecord]:
        rows = self._conn.execute(
            "SELECT store_id, product_id, available_qty, updated_at FROM stock "
            "ORDER BY store_id, product_id"
        )
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> StockRecord:
        store_id, product_id, qty, updated_at = row
        return StockRecord(store_id, product_id, qty, datetime.fromisoformat(updated_at))


class SqliteUserRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_many(self, users: Iterable[User]) -> None:
        rows = [(u.user_id, u.home_store_id) for u in users]
        with _write(self._conn):
            self._conn.executemany("INSERT INTO users VALUES (?,?)", rows)

    def get(self, user_id: str) -> User | None:
        row = self._conn.execute(
            "SELECT user_id, home_store_id FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return None if row is None else User(*row)

    def list_all(self) -> list[User]:
        rows = self._conn.execute("SELECT user_id, home_store_id FROM users ORDER BY user_id")
        return [User(*row) for row in rows]


class SqliteOrderRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_many(self, orders: Iterable[Order]) -> None:
        staged = list(orders)
        order_rows = [(o.order_id, o.user_id, o.store_id, _iso(o.placed_at)) for o in staged]
        line_rows = [
            (o.order_id, line_no, line.product_id, line.units)
            for o in staged
            for line_no, line in enumerate(o.lines, 1)
        ]
        with _write(self._conn):
            self._conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", order_rows)
            self._conn.executemany("INSERT INTO order_lines VALUES (?,?,?,?)", line_rows)

    def for_user(self, user_id: str) -> list[Order]:
        return self._load("WHERE o.user_id = ? ORDER BY o.placed_at, o.order_id", (user_id,))

    def list_all(self) -> list[Order]:
        return self._load("ORDER BY o.order_id", ())

    def _load(self, clause: str, params: tuple[object, ...]) -> list[Order]:
        rows = self._conn.execute(
            "SELECT o.order_id, o.user_id, o.store_id, o.placed_at, l.product_id, l.units "
            f"FROM orders o JOIN order_lines l USING (order_id) {clause}, l.line_no",
            params,
        )
        orders: list[Order] = []
        header: tuple[str, str, str, str] | None = None
        lines: list[OrderLine] = []
        for order_id, user_id, store_id, placed_at, product_id, units in rows:
            if header is not None and header[0] != order_id:
                orders.append(self._build(header, lines))
                lines = []
            header = (order_id, user_id, store_id, placed_at)
            lines.append(OrderLine(product_id, units))
        if header is not None:
            orders.append(self._build(header, lines))
        return orders

    @staticmethod
    def _build(header: tuple[str, str, str, str], lines: list[OrderLine]) -> Order:
        order_id, user_id, store_id, placed_at = header
        return Order(order_id, user_id, store_id, datetime.fromisoformat(placed_at), tuple(lines))
