"""Seeded simulation of stores, store inventory, users and order history.

Determinism: every entity draws from its own ``random.Random`` seeded with
``"{seed}:{stage}:{entity_id}"`` (string seeds are hashed with SHA-512, independent of
PYTHONHASHSEED). Adding users therefore never changes existing users' histories, and changing
user settings never changes inventory. Exact floats can differ across Python versions, which is
why snapshot manifests record the interpreter version.

Model (deliberately simple and documented, since personalization is evaluated against it):
- Global popularity is Zipfian over a seeded shuffle of the catalog.
- Each store carries a random share of the catalog; some carried products have zero stock.
  A product with no StockRecord is "not carried", distinct from "out of stock".
- Each user has a home store, sparse category affinities and one preferred brand per category,
  and buys from their home store's assortment, choosing the preferred brand with probability
  ``brand_loyalty`` and otherwise any brand, weighted by popularity.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from cartiq.config import SimulationSettings
from cartiq.domain import Order, OrderLine, Product, StockRecord, Store, User

_MAX_UNITS_PER_LINE = 3
_AFFINITY_SHAPE = 0.3  # gamma shape < 1 makes category interest sparse
_MIN_WEIGHT = 1e-9

Category = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimulatedWorld:
    stores: tuple[Store, ...]
    stock: tuple[StockRecord, ...]
    users: tuple[User, ...]
    orders: tuple[Order, ...]


def simulate(products: Sequence[Product], settings: SimulationSettings) -> SimulatedWorld:
    ordered = sorted(products, key=lambda p: p.product_id)
    stores = _stores(settings)
    popularity = _popularity(ordered, settings)
    stock = [record for store in stores for record in _stock(store, ordered, settings)]
    assortment = _assortments(stock, {p.product_id: p for p in ordered})

    users: list[User] = []
    orders: list[Order] = []
    for index in range(1, settings.n_users + 1):
        user_id = f"user-{index:05d}"
        rng = _rng(settings, "user", user_id)
        home = rng.choice(stores)
        users.append(User(user_id, home_store_id=home.store_id))
        orders.extend(_orders(user_id, home, assortment[home.store_id], popularity, settings, rng))

    return SimulatedWorld(tuple(stores), tuple(stock), tuple(users), tuple(orders))


def _rng(settings: SimulationSettings, stage: str, entity_id: str = "") -> random.Random:
    return random.Random(f"{settings.seed}:{stage}:{entity_id}")


def _stores(settings: SimulationSettings) -> list[Store]:
    stores = []
    for index in range(1, settings.n_stores + 1):
        city = settings.cities[(index - 1) % len(settings.cities)]
        stores.append(Store(f"store-{index:03d}", name=f"CartIQ {city} #{index}", city=city))
    return stores


def _popularity(products: Sequence[Product], settings: SimulationSettings) -> dict[str, float]:
    ids = [p.product_id for p in products]
    _rng(settings, "popularity").shuffle(ids)
    return {pid: 1.0 / rank**settings.popularity_exponent for rank, pid in enumerate(ids, 1)}


def _stock(
    store: Store, products: Sequence[Product], settings: SimulationSettings
) -> list[StockRecord]:
    rng = _rng(settings, "inventory", store.store_id)
    records = []
    for product in products:
        if rng.random() >= settings.assortment_ratio:
            continue
        out_of_stock = rng.random() < settings.out_of_stock_rate
        qty = 0 if out_of_stock else rng.randint(1, settings.max_stock_units)
        records.append(
            StockRecord(store.store_id, product.product_id, qty, settings.inventory_as_of)
        )
    return records


def _assortments(
    stock: Sequence[StockRecord], products: dict[str, Product]
) -> dict[str, dict[Category, list[Product]]]:
    """store_id -> leaf category -> carried products (sorted by id)."""
    by_store: dict[str, dict[Category, list[Product]]] = defaultdict(lambda: defaultdict(list))
    for record in stock:
        product = products[record.product_id]
        by_store[record.store_id][product.category_path].append(product)
    return by_store


def _orders(
    user_id: str,
    home: Store,
    assortment: dict[Category, list[Product]],
    popularity: dict[str, float],
    settings: SimulationSettings,
    rng: random.Random,
) -> list[Order]:
    categories = sorted(assortment)
    if not categories:
        return []
    affinity = [rng.gammavariate(_AFFINITY_SHAPE, 1.0) + _MIN_WEIGHT for _ in categories]
    preferred = {
        category: _preferred_brand(assortment[category], popularity, rng) for category in categories
    }

    n_orders = rng.randint(settings.min_orders_per_user, settings.max_orders_per_user)
    window_seconds = settings.history_days * 86_400
    start = datetime.combine(settings.history_start, time.min, tzinfo=UTC)
    offsets = sorted(rng.randrange(window_seconds) for _ in range(n_orders))

    orders = []
    for number, offset in enumerate(offsets, 1):
        units: dict[str, int] = {}
        for _ in range(rng.randint(1, settings.max_lines_per_order)):
            category = rng.choices(categories, weights=affinity)[0]
            choices = assortment[category]
            if rng.random() < settings.brand_loyalty:
                choices = [p for p in choices if p.brand == preferred[category]]
            weights = [popularity[p.product_id] for p in choices]
            product = rng.choices(choices, weights=weights)[0]
            units[product.product_id] = units.get(product.product_id, 0) + rng.randint(
                1, _MAX_UNITS_PER_LINE
            )
        orders.append(
            Order(
                order_id=f"order-{user_id.removeprefix('user-')}-{number:03d}",
                user_id=user_id,
                store_id=home.store_id,
                placed_at=start + timedelta(seconds=offset),
                lines=tuple(OrderLine(pid, qty) for pid, qty in sorted(units.items())),
            )
        )
    return orders


def _preferred_brand(
    products: Sequence[Product], popularity: dict[str, float], rng: random.Random
) -> str | None:
    """Pick a brand with probability proportional to its products' total popularity."""
    brand_weight: dict[str | None, float] = defaultdict(float)
    for product in products:
        brand_weight[product.brand] += popularity[product.product_id]
    brands = sorted(brand_weight, key=lambda b: (b is None, b or ""))
    return rng.choices(brands, weights=[brand_weight[b] for b in brands])[0]
