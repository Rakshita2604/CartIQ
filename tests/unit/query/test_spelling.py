import pytest

from cartiq.config import BM25Settings, QuerySettings
from cartiq.domain import CorrectionReason, Product
from cartiq.query.lexicon import load_lexicon
from cartiq.query.spelling import SpellingCorrector, osa_distance
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
