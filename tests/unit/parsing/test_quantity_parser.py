import pytest

from cartiq.domain import Quantity, Unit
from cartiq.parsing.quantity import parse_quantity


@pytest.mark.parametrize(
    ("text", "quantity", "pack_count"),
    [
        ("Amul Taaza Toned Milk 500 ml", Quantity(500, Unit.ML), 1),
        ("Aashirvaad Atta 5kg", Quantity(5, Unit.KG), 1),
        ("Fortune Sunflower Oil 1 Ltr", Quantity(1, Unit.L), 1),
        ("Sugar 1.5 KGS", Quantity(1.5, Unit.KG), 1),
        ("Tata Salt 1 kilo", Quantity(1, Unit.KG), 1),
        ("Tea 250 Gms", Quantity(250, Unit.G), 1),
        ("Garlic Capsule 500 mg", Quantity(0.5, Unit.G), 1),
        ("Ghee 1/2 litre", Quantity(0.5, Unit.L), 1),
        ("Eggs 6 pcs", Quantity(6, Unit.PCS), 1),
        ("Bananas 12 Nos", Quantity(12, Unit.PCS), 1),
        ("Coca-Cola 6 x 300 ml", Quantity(300, Unit.ML), 6),
        ("Frooti 6x200ml", Quantity(200, Unit.ML), 6),
        ("Frooti 200 ml x 6", Quantity(200, Unit.ML), 6),
        ("Frooti 200mlx6", Quantity(200, Unit.ML), 6),
        ("Maggi Noodles 70 g (Pack of 12)", Quantity(70, Unit.G), 12),
        ("Liquid Detergent 1kg (1kg Free)", Quantity(1, Unit.KG), 1),
        ("Kitchen Cleaner + Floor Cleaner (Each 450 ml)", Quantity(450, Unit.ML), 1),
        ("Dog Treat (10-25 kg), 500 g", Quantity(500, Unit.G), 1),
    ],
)
def test_parses_quantity(text: str, quantity: Quantity, pack_count: int) -> None:
    parsed = parse_quantity(text)

    assert parsed is not None
    assert parsed.quantity == quantity
    assert parsed.pack_count == pack_count


def test_pack_of_without_amount_has_unknown_unit_quantity() -> None:
    parsed = parse_quantity("Kitchen Sponge - Pack of 3")

    assert parsed is not None
    assert parsed.quantity is None
    assert parsed.pack_count == 3


@pytest.mark.parametrize(
    "text",
    [
        "Fresh Lemon",
        "1 lemon",
        "Vitamin B12 Tablets",
        "Brass Angle Deep - Plain, No.1",
        "2 in 1 Shampoo",
        "0 g sugar cola",
        "Rs 50 off",
        "",
    ],
)
def test_ignores_non_quantities(text: str) -> None:
    assert parse_quantity(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Honey 1 kg + Corn Flakes Classic, 475 g",
        "Horlicks 500 g Jar + 75 g Free",
        "Badam Tail - Almond Oil 100 ml + 50 ml",
        "Apple Shimla 4 pcs + Banana Robusta 1 kg",
    ],
)
def test_different_amounts_are_ambiguous(text: str) -> None:
    assert parse_quantity(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Wonder Diaper Pants - Xtra Large, 12-17 Kg",
        "Baby Dry Large (9 - 14 kg)",
        "Pearl Spot Fish - Medium, 3\u20134 pcs",
    ],
)
def test_ranges_are_not_quantities(text: str) -> None:
    assert parse_quantity(text) is None


def test_reports_span_of_mention() -> None:
    text = "amul milk 1l toned"
    parsed = parse_quantity(text)

    assert parsed is not None
    assert text[parsed.start : parsed.end] == "1l"
