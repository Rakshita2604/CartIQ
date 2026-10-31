import pytest

from cartiq.query.phonetic import phonetic_distance, soundex


@pytest.mark.parametrize(
    ("word", "code"),
    [
        # Standard American Soundex reference cases.
        ("Robert", "R163"),
        ("Rupert", "R163"),
        ("Ashcraft", "A261"),  # h does not separate s and c
        ("Tymczak", "T522"),  # a vowel lets k repeat 2
        ("Pfister", "P236"),  # f has the first letter's digit
        ("Honeyman", "H555"),
        # The cases that motivated the tie-break.
        ("magii", "M200"),
        ("maggi", "M200"),
        ("magic", "M220"),
        ("magix", "M220"),
        ("a", "A000"),
    ],
)
def test_soundex(word: str, code: str) -> None:
    assert soundex(word) == code


def test_non_letters_are_ignored() -> None:
    assert soundex("") == ""
    assert soundex("123") == ""
    assert soundex("café") == soundex("caf")


def test_phonetic_distance() -> None:
    assert phonetic_distance("magii", "maggi") == 0
    assert phonetic_distance("magii", "magic") == 1
