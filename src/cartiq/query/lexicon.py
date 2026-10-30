"""Load the frozen Hinglish lexicon and brand alias files.

Both files are frozen: their SHA-256 is pinned here, and loading refuses any file whose
content differs. A change means a new version file and a new pinned hash, never an edit.
"""

import hashlib
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from cartiq.parsing.text import tokenize

FROZEN_SHA256 = {
    "hinglish-v1": "6dbc855240ea42ef63f03b9fd59a3af903091ea47bf6c9bf1c60cdbbe6225881",
    "brand-aliases-v1": "3465705887b886d3dc73470d5934115e74e82949f0b3339b63108cfc93409500",
}
# Hindi connectors ("sarson ka tel"). Dropped only inside a matched lexicon phrase.
CONNECTORS = frozenset({"ka", "ki", "ke"})

Phrase = tuple[str, ...]


class LexiconError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    group: str
    terms: tuple[Phrase, ...]  # what a shopper may type
    english: tuple[Phrase, ...]  # catalog-language equivalents


@dataclass(frozen=True, slots=True)
class BrandAlias:
    alias: Phrase
    brand: str


@dataclass(frozen=True, slots=True)
class Lexicon:
    version: str
    sha256: str
    entries: tuple[LexiconEntry, ...]
    _phrases: dict[Phrase, LexiconEntry] = field(init=False, repr=False, compare=False)
    _longest: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        phrases: dict[Phrase, LexiconEntry] = {}
        for entry in self.entries:
            for phrase in entry.terms:
                if phrase in phrases:
                    raise LexiconError(f"{' '.join(phrase)!r} is in two lexicon entries")
                phrases[phrase] = entry
        object.__setattr__(self, "_phrases", phrases)
        object.__setattr__(self, "_longest", max((len(p) for p in phrases), default=0))

    @property
    def single_terms(self) -> frozenset[str]:
        """One-token typed terms (never spelling-corrected; spelling may correct to them)."""
        return frozenset(phrase[0] for phrase in self._phrases if len(phrase) == 1)

    def entry(self, phrase: Phrase) -> LexiconEntry | None:
        return self._phrases.get(phrase)

    def longest_match(self, tokens: list[str], start: int) -> tuple[int, LexiconEntry] | None:
        """The longest typed phrase starting at ``tokens[start]``, as (length, entry)."""
        for length in range(min(self._longest, len(tokens) - start), 0, -1):
            entry = self._phrases.get(tuple(tokens[start : start + length]))
            if entry is not None:
                return length, entry
        return None


def load_lexicon(version: str) -> Lexicon:
    document, sha256 = _load_frozen(version)
    entries = tuple(
        LexiconEntry(
            group=item["group"],
            terms=tuple(_phrase(term) for term in item["terms"]),
            english=tuple(_phrase(term) for term in item["english"]),
        )
        for item in document["entry"]
    )
    return Lexicon(version=version, sha256=sha256, entries=entries)


def load_brand_aliases(version: str) -> tuple[tuple[BrandAlias, ...], str]:
    document, sha256 = _load_frozen(version)
    aliases = tuple(BrandAlias(_phrase(item["alias"]), item["brand"]) for item in document["alias"])
    return aliases, sha256


def _load_frozen(version: str) -> tuple[dict[str, Any], str]:
    expected = FROZEN_SHA256.get(version)
    if expected is None:
        raise LexiconError(f"unknown lexicon version {version!r}")
    raw = resources.files("cartiq.query").joinpath("lexicon", f"{version}.toml").read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha256 != expected:
        raise LexiconError(f"{version} has sha256 {sha256}, frozen as {expected}")
    document = tomllib.loads(raw.decode("utf-8"))
    if document.get("schema_version") != 1 or document.get("version") != version:
        raise LexiconError(f"{version}: unexpected schema_version or version")
    return document, sha256


def _phrase(text: str) -> Phrase:
    phrase = tuple(tokenize(text))
    if not phrase:
        raise LexiconError(f"lexicon term {text!r} has no tokens")
    return phrase
