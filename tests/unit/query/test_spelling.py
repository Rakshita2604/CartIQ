import pytest

from cartiq.config import BM25Settings, QuerySettings
from cartiq.domain import CorrectionReason, Product
from cartiq.query.lexicon import load_lexicon
from cartiq.query.spelling import SpellingCorrector
from cartiq.query.spelling_distance import osa_distance
from cartiq.retrieval import BM25FIndex

SETTINGS = QuerySettings()


def corrector(df: dict[str, int], protected: tuple[str, ...] = ()) -> SpellingCorrector:
    return SpellingCorrector(df, protected, SETTINGS)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("milk", "milk", 0),
        ("amull", "amul", 1),  # deletion
        ("panner", "paneer", 1),  # substitution
        ("ab", "ba", 1),  # adjacent swap
        ("biscut", "biscuits", 2),
        ("kitten", "sitting", 3),
        ("", "abc", 3),
    ],
)
def test_osa_distance(a: str, b: str, distance: int) -> None:
    assert osa_distance(a, b) == distance == osa_distance(b, a)


class TestUnknownTokens:
    def test_nearest_then_most_frequent_then_alphabetical(self) -> None:
        speller = corrector({"maggi": 4, "magic": 3, "mango": 9})

        fix = speller.correct("magii")

        assert fix is not None
        assert (fix.corrected, fix.distance, fix.reason) == ("maggi", 1, CorrectionReason.UNKNOWN)
        assert corrector({"cart": 2, "card": 2}).correct("carx").corrected == "card"  # type: ignore[union-attr]

    def test_distance_allowed_depends_on_length(self) -> None:
        speller = corrector({"coriander": 3, "pepper": 2, "tea": 5})

        assert speller.correct("corriandr").corrected == "coriander"  # type: ignore[union-attr]  # 9 chars, 2 edits
        assert speller.correct("peper") is not None  # 5 chars, 1 edit
        assert osa_distance("pepxxr", "pepper") == 2
        assert speller.correct("pepxxr") is None  # 6 chars, 2 edits away: not allowed
        assert speller.correct("tee") is None  # under 4 chars: never corrected

    def test_lexicon_terms_are_targets_but_never_corrected(self) -> None:
        speller = corrector({"masala": 3}, protected=("dhaniya", "masaala"))

        assert speller.correct("dhanya").corrected == "dhaniya"  # type: ignore[union-attr]
        assert speller.correct("masaala") is None

    def test_tokens_with_digits_are_never_corrected(self) -> None:
        assert corrector({"500ml": 2}).correct("500mll") is None

    def test_no_candidate(self) -> None:
        assert corrector({"milk": 9}).correct("zzzzzz") is None


class TestDominance:
    def test_dominant_neighbour_of_a_rare_known_token_is_added(self) -> None:
        fix = corrector({"panner": 1, "paneer": 40, "banner": 3}).correct("panner")

        assert fix is not None
        assert (fix.corrected, fix.reason) == ("paneer", CorrectionReason.DOMINANT)

    @pytest.mark.parametrize(
        "df",
        [
            {"panner": 1, "paneer": 19},  # below 20x
            {"panner": 2, "paneer": 39},  # below 20x
            {"rare": 0, "care": 4},  # 20x satisfied but below the minimum df of 5
        ],
    )
    def test_not_dominant_enough(self, df: dict[str, int]) -> None:
        token = next(iter(df))

        assert corrector(df).correct(token) is None

    def test_exactly_at_the_thresholds(self) -> None:
        assert corrector({"panner": 1, "paneer": 20}).correct("panner") is not None
        assert corrector({"rare": 0, "care": 5}).correct("rare") is not None


@pytest.fixture(scope="module")
def synthetic_speller(synthetic_products: tuple[Product, ...]) -> SpellingCorrector:
    index = BM25FIndex.build(synthetic_products, BM25Settings())
    df = {term: index.document_frequency(term) for term in index.vocabulary}
    return SpellingCorrector(df, load_lexicon("hinglish-v1").single_terms, SETTINGS)


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("amull", "amul"),
        ("panner", "paneer"),
        ("tomatto", "tomato"),
        ("detergant", "detergent"),
        ("colgat", "colgate"),
        ("magii", "maggi"),
        ("corriander", "coriander"),
        ("bannana", "banana"),
        ("sunflowr", "sunflower"),
        ("biscut", "biskut"),  # nearest target is the lexicon spelling, which expands
    ],
)
def test_development_misspellings(
    synthetic_speller: SpellingCorrector, typed: str, expected: str
) -> None:
    fix = synthetic_speller.correct(typed)

    assert fix is not None
    assert fix.corrected == expected


# Fixed before running: valid words that must not be changed. Deliberately avoids every
# word of the held-out queries.
DO_NOT_CORRECT = ["chicken", "iphone", "dahi", "pizza", "vodka", "laptop", "garam", "masala"]


@pytest.mark.parametrize("word", DO_NOT_CORRECT)
def test_do_not_correct(synthetic_speller: SpellingCorrector, word: str) -> None:
    assert synthetic_speller.correct(word) is None


class TestSpellingVocabulary:
    @staticmethod
    def product(
        pid: str, name: str, brand: str | None, path: tuple[str, ...], desc: str
    ) -> Product:
        from decimal import Decimal

        return Product(pid, name, path, Decimal(10), brand=brand, description=desc)

    def test_counts_name_brand_and_product_type_only(self) -> None:
        from cartiq.query.spelling import spelling_frequencies

        products = [
            self.product(
                "P1",
                "Coca-Cola Coke Zero",
                "Coca-Cola",
                ("Beverages", "Soft Drinks", "Cold Drinks"),
                "Comes chilled; come and get it",
            ),
            self.product("P2", "Cold Coffee", None, ("Beverages", "Soft Drinks", "Coffee"), "come"),
        ]

        df = spelling_frequencies(products, ("name", "brand", "product_type"))

        assert df["coke"] == 1
        assert df["coca"] == 1  # brand field (also stripped from the name prefix)
        assert df["cold"] == 2  # P1 product type, P2 name
        assert df["drinks"] == 1
        assert "come" not in df  # description only
        assert "soft" not in df  # sub-category only
        assert "beverages" not in df  # top-level category

    def test_description_only_words_are_never_targets(self) -> None:
        from cartiq.query.spelling import spelling_frequencies

        products = [
            self.product(f"P{i}", "Coke Zero", "Coca-Cola", ("Drinks", "Cola"), "come home")
            for i in range(40)
        ]
        restricted = SpellingCorrector(
            spelling_frequencies(products, ("name", "brand", "product_type")), (), SETTINGS
        )
        everything = SpellingCorrector(
            spelling_frequencies(
                products, ("name", "brand", "product_type", "sub_category", "description")
            ),
            (),
            SETTINGS,
        )

        assert "come" in everything.candidates("cme")
        assert "come" not in restricted.candidates("cme")
        assert restricted.correct("comee") is None  # its only 1-edit target was "come"

    def test_candidates_are_the_unverified_lookup_set(self) -> None:
        speller = corrector({"maggi": 4, "magic": 3, "mango": 9})

        assert speller.candidates("magii") >= {"maggi", "magic"}
        assert speller.candidates("tee") == frozenset()  # too short: no lookup


class TestBrandFirstTieBreak:
    def test_brand_word_wins_among_equally_close_candidates(self) -> None:
        df = {"maggi": 4, "magic": 30}

        assert corrector(df).correct("magii").corrected == "magic"  # type: ignore[union-attr]
        preferring = SpellingCorrector(df, (), SETTINGS, preferred=("maggi",))
        assert preferring.correct("magii").corrected == "maggi"  # type: ignore[union-attr]

    def test_a_closer_non_brand_candidate_still_wins(self) -> None:
        df = {"tomato": 9, "tamatos": 1}
        speller = SpellingCorrector(df, (), SETTINGS, preferred=("tamatos",))

        assert speller.correct("tomatto").corrected == "tomato"  # type: ignore[union-attr]

    def test_dominance_rule_is_unchanged_by_preference(self) -> None:
        df = {"panner": 1, "paneer": 40, "panter": 40}
        speller = SpellingCorrector(df, (), SETTINGS, preferred=("panter",))

        assert speller.correct("panner").corrected == "paneer"  # type: ignore[union-attr]


MAGII_TARGETS = {"maggi": 33, "magic": 190, "magix": 41, "magik": 1}  # BigBasket counts


class TestPhoneticTieBreak:
    def test_phonetic_breaks_ties_after_brand_preference(self) -> None:
        all_brand = SpellingCorrector(
            MAGII_TARGETS, (), SETTINGS, preferred=("maggi", "magic", "magix"), phonetic=True
        )

        assert all_brand.correct("magii").corrected == "maggi"  # type: ignore[union-attr]

    def test_brand_preference_still_comes_first(self) -> None:
        only_magic = SpellingCorrector(
            MAGII_TARGETS, (), SETTINGS, preferred=("magic",), phonetic=True
        )

        assert only_magic.correct("magii").corrected == "magic"  # type: ignore[union-attr]

    def test_frequency_then_alphabet_follow(self) -> None:
        # "magic" and "magix" sound alike (M220): frequency decides; equal df: alphabetical.
        speller = SpellingCorrector({"magic": 5, "magix": 9}, (), SETTINGS, phonetic=True)
        assert speller.correct("magiq").corrected == "magix"  # type: ignore[union-attr]
        tied = SpellingCorrector({"magic": 5, "magix": 5}, (), SETTINGS, phonetic=True)
        assert tied.correct("magiq").corrected == "magic"  # type: ignore[union-attr]

    def test_off_by_default(self) -> None:
        assert SpellingCorrector(MAGII_TARGETS, (), SETTINGS).correct("magii").corrected == "magic"  # type: ignore[union-attr]

    def test_never_overrides_edit_distance(self) -> None:
        speller = SpellingCorrector(
            {"magii": 0, "maggies": 50, "mangi": 2}, (), SETTINGS, phonetic=True
        )

        assert speller.correct("magie").corrected == "magii"  # type: ignore[union-attr]
