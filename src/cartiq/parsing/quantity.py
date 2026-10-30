"""Extract pack quantities such as "500 g", "1.5 ltr" or "6 x 200 ml" from free text.

Shared by catalog ingestion (product names) and, later, query understanding.
Only numeric amounts are handled here; Hinglish number words come with query parsing.
"""

import re
from dataclasses import dataclass
from fractions import Fraction

from cartiq.domain import Quantity, Unit

# Alias -> (unit, multiplier into that unit). Milligrams are folded into grams.
_UNIT_ALIASES: dict[str, tuple[Unit, Fraction]] = {
    **dict.fromkeys(["mg", "mgs"], (Unit.G, Fraction(1, 1000))),
    **dict.fromkeys(["g", "gm", "gms", "gr", "gram", "grams"], (Unit.G, Fraction(1))),
    **dict.fromkeys(
        ["kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"], (Unit.KG, Fraction(1))
    ),
    **dict.fromkeys(
        ["ml", "mls", "millilitre", "millilitres", "milliliter", "milliliters"],
        (Unit.ML, Fraction(1)),
    ),
    **dict.fromkeys(
        ["l", "lt", "ltr", "ltrs", "litre", "litres", "liter", "liters"], (Unit.L, Fraction(1))
    ),
    **dict.fromkeys(["pc", "pcs", "piece", "pieces", "nos"], (Unit.PCS, Fraction(1))),
}

_UNIT = "|".join(sorted(map(re.escape, _UNIT_ALIASES), key=len, reverse=True))
_NUMBER = r"\d+/\d+|\d+(?:\.\d+)?"
# A number must not continue a word ("B12") or another number.
_NUMBER_START = r"(?<![\w.])"
# A unit must not run into a following letter ("1 lemon" is not one litre), except the
# multiplier in "200mlx6".
_UNIT_END = r"(?=[^a-z]|$|x\s*\d)"
_AMOUNT = rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT}){_UNIT_END}"
_TIMES = r"\s*[x\u00d7*]\s*"  # x, the multiplication sign, or *

_COUNT_FIRST = re.compile(rf"{_NUMBER_START}(?P<count>\d+){_TIMES}{_AMOUNT}")
_COUNT_LAST = re.compile(rf"{_NUMBER_START}{_AMOUNT}{_TIMES}(?P<count>\d+)(?![\w.])")
_SINGLE = re.compile(rf"{_NUMBER_START}{_AMOUNT}")
_PACK_OF = re.compile(r"\bpack\s+of\s+(?P<count>\d+)\b")
# Matched against the text just before an amount: "12-" in "12-17 kg" marks a range.
_RANGE_BEFORE = re.compile(r"\d\s*[-\u2013]\s*$")


@dataclass(frozen=True, slots=True)
class ParsedQuantity:
    quantity: Quantity | None
    pack_count: int
    start: int
    end: int


def parse_quantity(text: str) -> ParsedQuantity | None:
    """Return the quantity mentioned in ``text``, or None.

    Ranges ("12-17 kg") are not quantities. Text naming two or more *different* amounts, such
    as bundles ("Honey 1 kg + Corn Flakes 475 g") or offers ("500 g + 75 g free"), is
    ambiguous and yields None: an unknown size is safer than a wrong one for quantity intent.
    Multi-pack forms ("6 x 200 ml", "200 ml x 6") win over a bare amount. "Pack of N" alone
    gives a pack count with an unknown per-unit quantity.
    """
    lowered = text.lower()
    amounts = [
        (match, quantity)
        for match in _SINGLE.finditer(lowered)
        if not _RANGE_BEFORE.search(lowered, 0, match.start())
        and (quantity := _to_quantity(match["value"], match["unit"])) is not None
    ]
    if len({quantity.to_base() for _, quantity in amounts}) > 1:
        return None

    for pattern in (_COUNT_FIRST, _COUNT_LAST):
        match = pattern.search(lowered)
        if match:
            quantity = _to_quantity(match["value"], match["unit"])
            count = int(match["count"])
            if quantity is not None and count >= 1:
                return ParsedQuantity(quantity, count, match.start(), match.end())

    pack_of = _PACK_OF.search(lowered)
    pack_count = int(pack_of["count"]) if pack_of else 1
    if pack_count < 1:
        pack_of, pack_count = None, 1

    if amounts:
        first, quantity = amounts[0]
        return ParsedQuantity(quantity, pack_count, first.start(), first.end())
    if pack_of:
        return ParsedQuantity(None, pack_count, pack_of.start(), pack_of.end())
    return None


def _to_quantity(raw_value: str, raw_unit: str) -> Quantity | None:
    unit, multiplier = _UNIT_ALIASES[raw_unit]
    value = Fraction(raw_value) * multiplier  # Fraction accepts "1/2" and "1.5" exactly
    if value <= 0:
        return None
    return Quantity(float(value), unit)
