from decimal import Decimal

import pytest

from cartiq.domain import Product
from cartiq.query.brands import (
    BrandGazetteer,
    BrandMatchKind,
    WordStatistics,
    brand_key,
    word_statistics,
)
from cartiq.query.lexicon import BrandAlias


def gazetteer(
    brands: dict[str, int],
    anywhere: dict[str, int] | None = None,
    aliases: tuple[BrandAlias, ...] = (),
    min_share: float = 0.5,
    in_brand: dict[str, int] | None = None,
) -> BrandGazetteer:
    """``brands`` maps a brand spelling to its number of products. Unless given, each brand
    word is in the brand of exactly that brand's products."""
    listing = [brand for brand, count in brands.items() for _ in range(count)]
    if in_brand is None:
        in_brand = {}
        for brand, count in brands.items():
            for word in set(brand_key_words(brand)):
                in_brand[word] = in_brand.get(word, 0) + count
    return BrandGazetteer(listing, aliases, WordStatistics(in_brand, anywhere or {}), min_share)


def brand_key_words(brand: str) -> list[str]:
    from cartiq.parsing.text import tokenize

    return tokenize(brand)


def found(g: BrandGazetteer, query: str) -> list[tuple[str, BrandMatchKind]]:
    return [(m.display, m.kind) for m in g.search(query.split()).matches]


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Coca-Cola", "cocacola"),
        ("Haldiram's", "haldirams"),
        ("MAGGI", "maggi"),
        ("Head & Shoulders", "headshoulders"),
    ],
)
def test_brand_key(name: str, key: str) -> None:
    assert brand_key(name) == key


class TestMatching:
    G = gazetteer(
        {"Coca-Cola": 2, "Haldiram's": 3, "Surf Excel": 2, "Tata": 1, "Tata Salt": 4, "Lay's": 3},
        aliases=(
            BrandAlias(("coke",), "Coca-Cola"),
            BrandAlias(("surf",), "Surf Excel"),
            BrandAlias(("pepsi",), "Pepsi"),
        ),
    )

    def test_multi_word_and_joined_spellings(self) -> None:
        assert found(self.G, "coca cola 750 ml") == [("Coca-Cola", BrandMatchKind.BRAND)]
        assert found(self.G, "cocacola") == [("Coca-Cola", BrandMatchKind.BRAND)]

    def test_alias_and_missing_alias_target(self) -> None:
        assert found(self.G, "coke") == [("Coca-Cola", BrandMatchKind.ALIAS)]
        assert found(self.G, "pepsi") == []  # alias target not in this catalog

    def test_longest_span_wins(self) -> None:
        assert found(self.G, "surf excel matic") == [("Surf Excel", BrandMatchKind.BRAND)]
        assert found(self.G, "surf") == [("Surf Excel", BrandMatchKind.ALIAS)]
        assert found(self.G, "tata salt") == [("Tata Salt", BrandMatchKind.BRAND)]

    @pytest.mark.parametrize(
        ("query", "kind"),
        [
            ("haldirams", BrandMatchKind.BRAND),
            ("haldiram", BrandMatchKind.VARIANT),
            ("lays", BrandMatchKind.BRAND),
        ],
    )
    def test_possessive_and_plural_variants(self, query: str, kind: BrandMatchKind) -> None:
        assert len(found(self.G, query)) == 1
        assert found(self.G, query)[0][1] is kind

    def test_variants_need_four_characters(self) -> None:
        assert found(gazetteer({"Abs": 2}), "ab") == []

    def test_variant_to_a_plain_brand_with_trailing_s(self) -> None:
        g = gazetteer({"Haldirams": 3})

        assert found(g, "haldiram") == [("Haldirams", BrandMatchKind.VARIANT)]

    def test_display_is_the_most_common_spelling(self) -> None:
        g = gazetteer({"Head & shoulders": 2, "Head & Shoulders": 5})

        assert found(g, "head shoulders") == [("Head & Shoulders", BrandMatchKind.BRAND)]

    def test_skipped_positions_are_not_matched(self) -> None:
        assert self.G.search(["coca", "cola"], skip=[0]).matches == ()


class TestCommonWordGuard:
    def test_recognised_at_exactly_the_minimum_share(self) -> None:
        g = gazetteer({"Real": 5}, anywhere={"real": 10})

        search = g.search(["real", "juice"])

        assert [m.display for m in search.matches] == ["Real"]
        assert search.notes == (
            "'real' used as brand 'Real': 5/10 products containing 'real' have it in a "
            "brand (50% >= 50%)",
        )

    def test_common_word_brand_is_not_recognised_and_is_reported(self) -> None:
        g = gazetteer({"Real": 4}, anywhere={"real": 10})

        search = g.search(["real", "juice"])

        assert search.matches == ()
        assert search.notes == (
            "'real' not used as brand 'Real': 4/10 products containing 'real' have it in a "
            "brand (40% < 50%)",
        )

    def test_word_shared_by_several_brands_counts_as_brand_associated(self) -> None:
        # "tata": 1 product of brand Tata, 5 of Tata Sampann, 2 of Tata Tea; 8 contain it.
        g = gazetteer({"Tata": 1, "Tata Sampann": 5, "Tata Tea": 2}, anywhere={"tata": 8})

        assert found(g, "tata salt") == [("Tata", BrandMatchKind.BRAND)]

    @pytest.mark.parametrize("word", ["premium", "garden", "eco", "fresh"])
    def test_false_positives_on_ordinary_words(self, word: str) -> None:
        g = gazetteer({word.title(): 3}, anywhere={word: 40})

        assert found(g, f"{word} tea") == []

    def test_multi_word_brands_and_aliases_are_not_guarded(self) -> None:
        g = gazetteer(
            {"Mother Dairy": 1, "Coca-Cola": 1},
            anywhere={"mother": 90, "dairy": 90, "coca": 1},
            aliases=(BrandAlias(("coke",), "Coca-Cola"),),
        )

        assert found(g, "mother dairy") == [("Mother Dairy", BrandMatchKind.BRAND)]
        assert found(g, "coke") == [("Coca-Cola", BrandMatchKind.ALIAS)]

    def test_possessive_brand_counts_as_one_word(self) -> None:
        g = gazetteer({"Haldiram's": 1}, anywhere={"haldiram": 5})

        assert found(g, "haldiram") == []


def test_word_statistics_count_products_not_occurrences() -> None:
    def product(pid: str, name: str, brand: str | None, description: str) -> Product:
        return Product(
            pid, name, ("Food", "Juices"), Decimal(10), brand=brand, description=description
        )

    stats = word_statistics(
        [
            product("P1", "Real Mango Juice", "Real", "real real fruit"),
            product("P2", "Mango Juice", None, "made with real fruit"),
            product("P3", "Tata Tea Gold", "Tata Tea", ""),
        ]
    )

    assert (stats.in_brand["real"], stats.anywhere["real"]) == (1, 2)
    assert (stats.in_brand["tata"], stats.anywhere["tata"]) == (1, 1)
    assert stats.anywhere["juices"] == 3  # category levels count as "anywhere"
    assert "mango" not in stats.in_brand


def test_protected_and_preferred_terms() -> None:
    g = gazetteer({"Coca-Cola": 1, "Aroma Magic": 1}, aliases=(BrandAlias(("coke",), "Coca-Cola"),))

    assert g.brand_terms == {"coca", "cola", "aroma", "magic"}
    assert g.protected_terms == {"coca", "cola", "aroma", "magic", "coke"}
