import hashlib
from importlib import resources

import pytest

from cartiq.query import lexicon as lexicon_module
from cartiq.query.lexicon import FROZEN_SHA256, LexiconError, load_brand_aliases, load_lexicon


def test_frozen_hashes_are_the_committed_ones() -> None:
    assert FROZEN_SHA256 == {
        "hinglish-v1": "6dbc855240ea42ef63f03b9fd59a3af903091ea47bf6c9bf1c60cdbbe6225881",
        "brand-aliases-v1": "3465705887b886d3dc73470d5934115e74e82949f0b3339b63108cfc93409500",
    }
    for version, expected in FROZEN_SHA256.items():
        raw = resources.files("cartiq.query").joinpath("lexicon", f"{version}.toml").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected


def test_loads_entries_and_phrases() -> None:
    lexicon = load_lexicon("hinglish-v1")

    assert len(lexicon.entries) == 123
    assert sum(len(entry.terms) for entry in lexicon.entries) == 238
    assert {"doodh", "pyaz", "gud"} <= lexicon.single_terms
    assert "sarson" in lexicon.single_terms
    assert not {"ka", "patta"} & lexicon.single_terms  # words of phrases are not terms


def test_longest_phrase_wins() -> None:
    lexicon = load_lexicon("hinglish-v1")
    tokens = ["sarson", "ka", "tel", "1", "l"]

    match = lexicon.longest_match(tokens, 0)

    assert match is not None
    length, entry = match
    assert length == 3
    assert entry.english == (("mustard", "oil"),)
    assert lexicon.longest_match(["sarson"], 0)[1].english == (("mustard",),)  # type: ignore[index]
    assert lexicon.longest_match(["milk"], 0) is None


def test_brand_aliases_load() -> None:
    aliases, sha256 = load_brand_aliases("brand-aliases-v1")

    assert len(aliases) == 9
    assert sha256 == FROZEN_SHA256["brand-aliases-v1"]
    assert {alias.brand for alias in aliases if alias.alias == ("coke",)} == {"Coca-Cola"}


def test_refuses_a_file_that_differs_from_its_frozen_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(lexicon_module.FROZEN_SHA256, "hinglish-v1", "0" * 64)

    with pytest.raises(LexiconError, match="frozen as"):
        load_lexicon("hinglish-v1")


def test_refuses_unknown_versions() -> None:
    with pytest.raises(LexiconError, match="unknown lexicon version"):
        load_lexicon("hinglish-v9")
