import math

import pytest

from cartiq.domain import Dimension, Quantity, Unit


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        (Quantity(1, Unit.KG), Quantity(1000, Unit.G)),
        (Quantity(1.5, Unit.L), Quantity(1500, Unit.ML)),
        (Quantity(500, Unit.G), Quantity(500, Unit.G)),
        (Quantity(6, Unit.PCS), Quantity(6, Unit.PCS)),
    ],
)
def test_to_base(quantity: Quantity, expected: Quantity) -> None:
    assert quantity.to_base() == expected


@pytest.mark.parametrize(
    ("unit", "dimension"),
    [
        (Unit.G, Dimension.MASS),
        (Unit.KG, Dimension.MASS),
        (Unit.ML, Dimension.VOLUME),
        (Unit.L, Dimension.VOLUME),
        (Unit.PCS, Dimension.COUNT),
    ],
)
def test_every_unit_has_a_dimension(unit: Unit, dimension: Dimension) -> None:
    assert Quantity(1, unit).dimension is dimension


def test_same_amount_across_units() -> None:
    assert Quantity(1, Unit.KG).same_amount(Quantity(1000, Unit.G))
    assert Quantity(0.5, Unit.L).same_amount(Quantity(500, Unit.ML))
    assert not Quantity(1, Unit.KG).same_amount(Quantity(900, Unit.G))


def test_incomparable_dimensions_are_never_the_same_amount() -> None:
    assert not Quantity(1, Unit.L).is_comparable(Quantity(1, Unit.KG))
    assert not Quantity(1000, Unit.ML).same_amount(Quantity(1000, Unit.G))


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_rejects_non_positive_or_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"Quantity\.value"):
        Quantity(value, Unit.G)
