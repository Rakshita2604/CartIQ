from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

import pytest

from cartiq.config import SimulationSettings
from cartiq.domain import Product
from cartiq.simulation import SimulatedWorld, simulate


@pytest.fixture
def world(
    synthetic_products: tuple[Product, ...], small_simulation: SimulationSettings
) -> SimulatedWorld:
    return simulate(synthetic_products, small_simulation)


def test_same_seed_same_world(
    synthetic_products: tuple[Product, ...], small_simulation: SimulationSettings
) -> None:
    assert simulate(synthetic_products, small_simulation) == simulate(
        synthetic_products, small_simulation
    )


def test_input_order_does_not_matter(
    synthetic_products: tuple[Product, ...], small_simulation: SimulationSettings
) -> None:
    shuffled = tuple(reversed(synthetic_products))

    assert simulate(shuffled, small_simulation) == simulate(synthetic_products, small_simulation)


def test_different_seed_different_world(
    synthetic_products: tuple[Product, ...], small_simulation: SimulationSettings
) -> None:
    other = small_simulation.model_copy(update={"seed": 7})

    assert simulate(synthetic_products, other) != simulate(synthetic_products, small_simulation)


def test_adding_users_keeps_existing_histories_and_inventory(
    synthetic_products: tuple[Product, ...],
    small_simulation: SimulationSettings,
    world: SimulatedWorld,
) -> None:
    bigger = simulate(synthetic_products, small_simulation.model_copy(update={"n_users": 80}))

    assert bigger.stock == world.stock
    assert bigger.users[: len(world.users)] == world.users
    existing = {u.user_id for u in world.users}
    assert tuple(o for o in bigger.orders if o.user_id in existing) == world.orders


def test_stock_respects_assortment_and_out_of_stock_rates(
    synthetic_products: tuple[Product, ...],
) -> None:
    settings = SimulationSettings(
        n_stores=4, n_users=0, assortment_ratio=0.6, out_of_stock_rate=0.2
    )
    stock = simulate(synthetic_products, settings).stock

    carried_share = len(stock) / (len(synthetic_products) * settings.n_stores)
    oos_share = sum(not r.in_stock for r in stock) / len(stock)
    assert carried_share == pytest.approx(0.6, abs=0.06)
    assert oos_share == pytest.approx(0.2, abs=0.06)
    assert all(r.updated_at == settings.inventory_as_of for r in stock)
    assert all(0 <= r.available_qty <= settings.max_stock_units for r in stock)


def test_orders_only_use_products_carried_at_home_store(world: SimulatedWorld) -> None:
    carried = {(r.store_id, r.product_id) for r in world.stock}
    home = {u.user_id: u.home_store_id for u in world.users}

    assert world.orders
    for order in world.orders:
        assert order.store_id == home[order.user_id]
        assert all((order.store_id, line.product_id) in carried for line in order.lines)


def test_order_timestamps_fall_in_history_window(
    world: SimulatedWorld, small_simulation: SimulationSettings
) -> None:
    start = datetime.combine(small_simulation.history_start, datetime.min.time(), tzinfo=UTC)
    end = start + timedelta(days=small_simulation.history_days)

    assert all(start <= o.placed_at < end for o in world.orders)


def test_order_counts_respect_bounds(
    world: SimulatedWorld, small_simulation: SimulationSettings
) -> None:
    per_user = Counter(o.user_id for o in world.orders)

    assert all(
        small_simulation.min_orders_per_user <= n <= small_simulation.max_orders_per_user
        for n in per_user.values()
    )
    assert all(1 <= len(o.lines) <= small_simulation.max_lines_per_order for o in world.orders)


def test_brand_loyalty_concentrates_purchases(synthetic_products: tuple[Product, ...]) -> None:
    def top_brand_share(loyalty: float) -> float:
        settings = SimulationSettings(n_users=60, brand_loyalty=loyalty, seed=3)
        brand_of = {p.product_id: p.brand for p in synthetic_products}
        category_of = {p.product_id: p.category_path for p in synthetic_products}
        counts: dict[tuple[str, tuple[str, ...]], Counter[str | None]] = defaultdict(Counter)
        for order in simulate(synthetic_products, settings).orders:
            for line in order.lines:
                key = (order.user_id, category_of[line.product_id])
                counts[key][brand_of[line.product_id]] += 1
        top = sum(c.most_common(1)[0][1] for c in counts.values())
        return top / sum(sum(c.values()) for c in counts.values())

    assert top_brand_share(0.95) > top_brand_share(0.0)
