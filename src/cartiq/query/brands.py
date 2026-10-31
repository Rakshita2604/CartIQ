"""Recognise brands named in a query, from the catalog's brand field and frozen aliases.

Brands are matched by key: their tokens joined ("Coca-Cola" -> "cocacola", "Haldiram's" ->
"haldirams", "MAGGI" -> "maggi"), so case, punctuation and spacing do not matter. A query
span matches, longest span first, by:

1. brand key: the span's key equals a catalog brand's key;
2. alias: the span's tokens equal a frozen alias ("coke" -> Coca-Cola), if that brand is in
   this catalog;
3. variant: the span's key plus or minus a final "s" equals a brand key, for keys of at
   least 4 characters (possessives and plurals: "haldiram" ~ "Haldiram's" / "Haldirams").

Common-word guard: a one-word brand (ignoring a possessive "'s") found by key or variant is
recognised only if its word is mostly a brand word in this catalog:

    brand share = products whose brand contains the word / products containing the word
                  anywhere (name, brand, category levels, description)

and brand share >= ``min_share``. A word in several brands counts as brand-associated in
each ("tata" in Tata, Tata Sampann, Tata Tea). Every decision, accepted or rejected, is
reported in ``notes`` with its numerator, denominator and share. Multi-word brands and
aliases are always recognised. Counts come from the catalog snapshot (``word_statistics``).
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from cartiq.domain import Product
from cartiq.parsing.text import tokenize
from cartiq.query.lexicon import BrandAlias
from cartiq.retrieval.documents import Field, product_fields

_VARIANT_MIN_LENGTH = 4


class BrandMatchKind(StrEnum):
    BRAND = "brand"
    ALIAS = "alias"
    VARIANT = "variant"


@dataclass(frozen=True, slots=True)
class BrandMatch:
    start: int  # token span in the searched token list
    end: int
    key: str
    display: str
    kind: BrandMatchKind
    brand_words: tuple[str, ...]  # the catalog spelling's words, without a possessive "s"


@dataclass(frozen=True, slots=True)
class WordStatistics:
    in_brand: Mapping[str, int]  # products whose brand field contains the word
    anywhere: Mapping[str, int]  # products containing the word in any field


def word_statistics(products: Iterable[Product]) -> WordStatistics:
    in_brand: Counter[str] = Counter()
    anywhere: Counter[str] = Counter()
    for product in products:
        add_word_statistics(product_fields(product), in_brand, anywhere)
    return WordStatistics(in_brand=dict(in_brand), anywhere=dict(anywhere))


def add_word_statistics(
    fields: Mapping[Field, Sequence[str]], in_brand: Counter[str], anywhere: Counter[str]
) -> None:
    """Count one tokenized product (``product_fields``) into the guard statistics."""
    in_brand.update(set(fields[Field.BRAND]))
    anywhere.update({token for tokens in fields.values() for token in tokens})


@dataclass(frozen=True, slots=True)
class BrandSearch:
    matches: tuple[BrandMatch, ...]
    notes: tuple[str, ...]


class BrandGazetteer:
    def __init__(
        self,
        brands: Iterable[str],
        aliases: Iterable[BrandAlias],
        words: WordStatistics,
        min_share: float,
    ) -> None:
        """``brands`` holds one entry per product (so counts are product counts)."""
        products_per_key: Counter[str] = Counter()
        spellings: dict[str, Counter[str]] = {}
        for brand in brands:
            key = brand_key(brand)
            if key:
                products_per_key[key] += 1
                spellings.setdefault(key, Counter())[brand] += 1
        # Most common catalog spelling, ties alphabetically.
        self._display = {
            key: min(counts, key=lambda s: (-counts[s], s)) for key, counts in spellings.items()
        }
        self._tokens = {key: tuple(tokenize(name)) for key, name in self._display.items()}
        self._words = {key: _words(tokens) for key, tokens in self._tokens.items()}
        self._products = products_per_key
        self._aliases = {
            alias.alias: brand_key(alias.brand)
            for alias in aliases
            if brand_key(alias.brand) in self._display
        }
        self._words_stats = words
        self._min_share = min_share
        self._longest = max(
            [len(t) for t in self._tokens.values()] + [len(a) for a in self._aliases], default=0
        )

    @property
    def brand_terms(self) -> frozenset[str]:
        """Every token of every catalog brand: preferred by the spelling tie-break."""
        return frozenset(token for tokens in self._tokens.values() for token in tokens)

    @property
    def protected_terms(self) -> frozenset[str]:
        """Brand and alias tokens: never spelling-corrected."""
        return self.brand_terms | {token for alias in self._aliases for token in alias}

    def display(self, key: str) -> str:
        return self._display[key]

    def search(self, tokens: Sequence[str], skip: Iterable[int] = ()) -> BrandSearch:
        """Brand mentions in ``tokens``, longest span first, ignoring positions in ``skip``."""
        blocked = set(skip)
        matches: list[BrandMatch] = []
        notes: list[str] = []
        position = 0
        while position < len(tokens):
            found = None
            for length in range(min(self._longest, len(tokens) - position), 0, -1):
                span = range(position, position + length)
                if blocked & set(span):
                    continue
                found = self._match(tuple(tokens[position : position + length]), position, notes)
                if found is not None:
                    break
            if found is None:
                position += 1
                continue
            matches.append(found)
            position = found.end
        return BrandSearch(tuple(matches), tuple(dict.fromkeys(notes)))

    def _match(self, span: tuple[str, ...], start: int, notes: list[str]) -> BrandMatch | None:
        key = "".join(span)
        end = start + len(span)
        if key in self._display:
            return self._guarded(span, key, start, end, BrandMatchKind.BRAND, notes)
        alias = self._aliases.get(span)
        if alias is not None:
            return self._build(alias, start, end, BrandMatchKind.ALIAS)
        if len(key) >= _VARIANT_MIN_LENGTH:
            for variant in (key + "s", key[:-1] if key.endswith("s") else None):
                if variant and variant in self._display:
                    return self._guarded(span, variant, start, end, BrandMatchKind.VARIANT, notes)
        return None

    def _guarded(
        self,
        span: tuple[str, ...],
        key: str,
        start: int,
        end: int,
        kind: BrandMatchKind,
        notes: list[str],
    ) -> BrandMatch | None:
        words = self._words[key]
        if len(words) == 1 and len(span) == 1:
            word = words[0]
            in_brand = self._words_stats.in_brand.get(word, 0)
            anywhere = max(self._words_stats.anywhere.get(word, 0), in_brand)
            share = in_brand / anywhere if anywhere else 1.0
            decision = "used as" if share >= self._min_share else "not used as"
            notes.append(
                f"{span[0]!r} {decision} brand {self._display[key]!r}: {in_brand}/{anywhere} "
                f"products containing {word!r} have it in a brand ({share:.0%} "
                f"{'>=' if share >= self._min_share else '<'} {self._min_share:.0%})"
            )
            if share < self._min_share:
                return None
        return self._build(key, start, end, kind)

    def _build(self, key: str, start: int, end: int, kind: BrandMatchKind) -> BrandMatch:
        return BrandMatch(start, end, key, self._display[key], kind, self._words[key])


def brand_key(text: str) -> str:
    return "".join(tokenize(text))


def _words(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Brand tokens without the lone "s" a possessive leaves ("haldiram", "s")."""
    words = tuple(token for token in tokens if token != "s")
    return words or tokens
