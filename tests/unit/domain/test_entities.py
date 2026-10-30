from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from cartiq.domain import (
    Candidate,
    Constraint,
    Grade,
    Judgment,
    MatchTier,
    Order,
    OrderLine,
    ParsedQuery,
    Product,
    Quantity,
    QuerySegment,
    ScoredProduct,
    StockRecord,
    Substitute,
    SubstituteReason,
    Unit,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_product(**overrides: Any) -> Product:
    fields: dict[str, Any] = {
        "product_id": "P1",
        "name": "Amul Taaza Toned Milk",
        "category_path": ("dairy", "milk"),
        "price": Decimal("54.00"),
        "brand": "amul",
        "quantity": Quantity(500, Unit.ML),
        "mrp": Decimal("56.00"),
    }
    fields.update(overrides)
    return Product(**fields)


class TestProduct:
    def test_valid_product(self) -> None:
        product = make_product()

        assert product.total_quantity == Quantity(500, Unit.ML)

    def test_total_quantity_multiplies_pack_count(self) -> None:
        product = make_product(quantity=Quantity(200, Unit.ML), pack_count=6)

        assert product.total_quantity == Quantity(1200, Unit.ML)

    def test_total_quantity_is_none_without_quantity(self) -> None:
        assert make_product(quantity=None).total_quantity is None

    def test_is_immutable(self) -> None:
        product = make_product()
        with pytest.raises(FrozenInstanceError):
            product.price = Decimal(1)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"product_id": ""},
            {"product_id": " P1"},
            {"name": "  "},
            {"category_path": ()},
            {"category_path": ("dairy", "")},
            {"brand": ""},
            {"price": Decimal(0)},
            {"mrp": Decimal("50.00")},
            {"pack_count": 0},
        ],
    )
    def test_rejects_invalid_fields(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="Product"):
            make_product(**overrides)


class TestStockRecord:
    def test_in_stock(self) -> None:
        record = StockRecord("S1", "P1", available_qty=3, updated_at=NOW)

        assert record.in_stock
        assert not replace(record, available_qty=0).in_stock

    def test_rejects_negative_stock(self) -> None:
        with pytest.raises(ValueError, match="available_qty"):
            StockRecord("S1", "P1", available_qty=-1, updated_at=NOW)

    def test_rejects_naive_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            StockRecord("S1", "P1", available_qty=1, updated_at=datetime(2026, 1, 1))  # noqa: DTZ001


class TestOrder:
    def test_requires_lines(self) -> None:
        with pytest.raises(ValueError, match="lines"):
            Order("O1", "U1", "S1", placed_at=NOW, lines=())

    def test_rejects_zero_units(self) -> None:
        with pytest.raises(ValueError, match="units"):
            OrderLine("P1", units=0)


class TestParsedQuery:
    def test_constraints_reflect_explicit_intent(self) -> None:
        query = ParsedQuery(
            raw="amul doodh 1l",
            normalized="amul milk 1l",
            tokens=("amul", "milk", "1l"),
            brand="amul",
            quantity=Quantity(1, Unit.L),
        )

        assert query.constraints == {Constraint.BRAND, Constraint.QUANTITY}

    def test_no_constraints_for_plain_query(self) -> None:
        assert ParsedQuery(raw="milk", normalized="milk", tokens=("milk",)).constraints == set()


class TestCandidate:
    def test_rejects_zero_rank(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            Candidate("P1", fused_score=0.1, bm25_rank=0)

    def test_rejects_non_finite_score(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Candidate("P1", fused_score=float("nan"))


class TestScoredProduct:
    def test_relaxed_must_name_relaxed_constraints(self) -> None:
        with pytest.raises(ValueError, match="RELAXED"):
            ScoredProduct("P1", score=1.0, tier=MatchTier.RELAXED)

    def test_exact_cannot_be_relaxed(self) -> None:
        with pytest.raises(ValueError, match="EXACT"):
            ScoredProduct(
                "P1", score=1.0, tier=MatchTier.EXACT, relaxed=frozenset({Constraint.BRAND})
            )

    def test_relaxed_with_constraints_is_valid(self) -> None:
        result = ScoredProduct(
            "P1", score=1.0, tier=MatchTier.RELAXED, relaxed=frozenset({Constraint.QUANTITY})
        )

        assert result.relaxed == {Constraint.QUANTITY}


def test_substitute_cannot_reference_itself() -> None:
    with pytest.raises(ValueError, match="itself"):
        Substitute("P1", "P1", score=0.9, reason=SubstituteReason.SAME_CATEGORY_SIZE)


class TestJudgment:
    def test_valid_judgment(self) -> None:
        judgment = Judgment("q1", "doodh", QuerySegment.HINGLISH, "P1", Grade.EXACT, "manual:v1")

        assert judgment.grade == 3

    def test_requires_grade_enum(self) -> None:
        with pytest.raises(ValueError, match="Grade"):
            Judgment("q1", "doodh", QuerySegment.HINGLISH, "P1", 5, "manual:v1")  # type: ignore[arg-type]

    def test_requires_labeler(self) -> None:
        with pytest.raises(ValueError, match="labeler"):
            Judgment("q1", "doodh", QuerySegment.HINGLISH, "P1", Grade.EXACT, "")
