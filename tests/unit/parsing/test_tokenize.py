import unicodedata

import pytest

from cartiq.parsing.text import tokenize


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Amul MILK", ["amul", "milk"]),
        ("amul milk", ["amul", "milk"]),
        ("Lay's India's Magic Masala", ["lay", "s", "india", "s", "magic", "masala"]),
        ("Atta, Flours & Sooji", ["atta", "flours", "sooji"]),
        ("Coca-Cola 6x300ml", ["coca", "cola", "6x300ml"]),
        ("snake_case/slash", ["snake", "case", "slash"]),
        ("100% Pure", ["100", "pure"]),
        ("  spaced\tout\n", ["spaced", "out"]),
    ],
)
def test_case_and_punctuation(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_composed_and_decomposed_accents_match() -> None:
    composed = unicodedata.normalize("NFC", "Café Nescafé")
    decomposed = unicodedata.normalize("NFD", "Café Nescafé")

    assert composed != decomposed
    assert tokenize(composed) == tokenize(decomposed) == ["café", "nescafé"]


def test_compatibility_forms_and_casefold() -> None:
    assert tokenize("\uff2d\uff29\uff2c\uff2b \uff15\uff10\uff10") == ["milk", "500"]  # full-width
    assert tokenize("ﬁne STRASSE straße") == ["fine", "strasse", "strasse"]


def test_duplicates_and_order_are_kept() -> None:
    assert tokenize("milk Milk MILK tea") == ["milk", "milk", "milk", "tea"]


@pytest.mark.parametrize("text", ["", "   ", "!!! & --- ...", "___"])
def test_empty_input_gives_no_tokens(text: str) -> None:
    assert tokenize(text) == []
