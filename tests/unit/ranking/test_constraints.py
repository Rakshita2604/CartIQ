from decimal import Decimal

import pytest

from cartiq.domain import (
    Candidate,
    Constraint,
    MatchTier,
    ParsedQuery,
    Product,
    Quantity,
    QuantityKind,
    QuantityMention,
    Unit,
)
from cartiq.ranking import order, quantity_satisfied, violations

ONE_LITRE = Quantity(1, Unit.L)


def product(
    pid: str = "P1",
    brand: str | None = "Amul",
    quantity: Quantity | None = ONE_LITRE,
    pack_count: int = 1,
) -> Product:
    return Product(
        pid,
        f"{brand or ''} Milk",
        ("Dairy", "Milk"),
        Decimal(10),
        brand=brand,
        quantity=quantity,
        pack_count=pack_count,
    )


def amount(value: float, unit: Unit, pack: int | None = None) -> QuantityMention:
    return QuantityMention(
        QuantityKind.AMOUNT, ((0, 4),), "x", quantity=Quantity(value, unit), pack_count=pack
    )


PACK_12 = QuantityMention(QuantityKind.PACK, ((0, 4),), "12 pack", pack_count=12)
BARE_12 = QuantityMention(QuantityKind.BARE_NUMBER, ((0, 2),), "12", count=12)


class TestQuantity:
    @pytest.mark.parametrize(
        ("mention", "item", "ok"),
        [
            (amount(1, Unit.L), product(quantity=Quantity(1000, Unit.ML)), True),  # 1 l = 1000 ml
            (amount(500, Unit.G), product(quantity=Quantity(500, Unit.ML)), False),  # never g = ml
            (amount(1, Unit.L), product(quantity=Quantity(500, Unit.ML)), False),
            (amount(1, Unit.L), product(quantity=None), False),  # unknown size violates
            # A size without a pack count accepts any pack of that per-unit size ...
            (amount(300, Unit.ML), product(quantity=Quantity(300, Unit.ML), pack_count=6), True),
            # ... an explicit pack count must match ...
            (
                amount(300, Unit.ML, pack=6),
                product(quantity=Quantity(300, Unit.ML), pack_count=6),
                True,
            ),
            (
                amount(300, Unit.ML, pack=6),
                product(quantity=Quantity(300, Unit.ML), pack_count=1),
                False,
            ),
            # ... and the pack total is never compared: 6 x 200 ml is not 1200 ml.
            (amount(1200, Unit.ML), product(quantity=Quantity(200, Unit.ML), pack_count=6), False),
            (PACK_12, product(quantity=Quantity(70, Unit.G), pack_count=12), True),
            (PACK_12, product(quantity=Quantity(70, Unit.G), pack_count=1), False),
            (PACK_12, product(quantity=None), False),  # default pack count of 1
            (BARE_12, product(quantity=Quantity(12, Unit.PCS)), True),  # 12 pieces
            (BARE_12, product(quantity=Quantity(70, Unit.G), pack_count=12), True),  # pack of 12
            (BARE_12, product(quantity=Quantity(6, Unit.PCS)), False),
            (BARE_12, product(quantity=None), False),
        ],
    )
    def test_quantity_satisfied(self, mention: QuantityMention, item: Product, ok: bool) -> None:
        assert quantity_satisfied(mention, item) is ok


def query(brand_key: str | None = None, mention: QuantityMention | None = None) -> ParsedQuery:
    return ParsedQuery(
        raw="q",
        normalized="q",
        tokens=("q",),
        brand_key=brand_key,
        brand=brand_key,
        quantity_mention=mention,
    )


class TestViolations:
    def test_no_constraints_and_full_coverage_is_exact(self) -> None:
        assert violations(query(), product(), {"milk"}, {"milk"}) == frozenset()

    def test_brand_by_key_and_missing_brand(self) -> None:
        q = query(brand_key="haldirams")
        possessive = product(brand="Haldiram's")
        plain = product(brand="Haldirams")

        assert violations(q, possessive, set(), set()) == frozenset()
        assert violations(q, plain, set(), set()) == frozenset()
        assert violations(q, product(brand="Bikaji"), set(), set()) == {Constraint.BRAND}
        assert violations(q, product(brand=None), set(), set()) == {Constraint.BRAND}

    def test_partial_coverage_across_several_concepts(self) -> None:
        required = {"basmati", "rice"}

        assert violations(query(), product(), {"basmati", "rice"}, required) == frozenset()
        assert violations(query(), product(), {"rice"}, required) == {Constraint.PRODUCT}
        assert violations(query(), product(), set(), required) == {Constraint.PRODUCT}

    def test_all_violations_are_listed(self) -> None:
        q = query(brand_key="amul", mention=amount(1, Unit.L))
        wrong = product(brand="Nandini", quantity=Quantity(500, Unit.ML))

        assert violations(q, wrong, {"milk"}, {"milk", "toned"}) == {
            Constraint.BRAND,
            Constraint.QUANTITY,
            Constraint.PRODUCT,
        }


class TestOrder:
    def test_exact_first_bm25_order_kept_and_scores_untouched(self) -> None:
        items = {
            "a": product("a", brand="Nandini"),
            "b": product("b", brand="Amul"),
            "c": product("c", brand=None),
            "d": product("d", brand="Amul"),
        }
        candidates = [
            Candidate("a", fused_score=9.0, bm25_score=9.0, bm25_rank=1),
            Candidate("b", fused_score=5.0, bm25_score=5.0, bm25_rank=2),
            Candidate("c", fused_score=4.0, bm25_score=4.0, bm25_rank=3),
            Candidate("d", fused_score=1.0, bm25_score=1.0, bm25_rank=37),
        ]
        coverage = dict.fromkeys(items, frozenset({"milk"}))

        scored = order(candidates, items, query(brand_key="amul"), coverage, {"milk"})

        assert [s.product_id for s in scored] == ["b", "d", "a", "c"]  # rank 37 promoted
        assert [s.tier for s in scored] == [MatchTier.EXACT] * 2 + [MatchTier.RELAXED] * 2
        assert [s.score for s in scored] == [5.0, 1.0, 9.0, 4.0]
        assert [s.features["bm25_rank"] for s in scored] == [2.0, 37.0, 1.0, 3.0]
        assert [s.relaxed for s in scored[2:]] == [{Constraint.BRAND}, {Constraint.BRAND}]

    def test_empty_candidates(self) -> None:
        assert order([], {}, query(), {}, set()) == []
