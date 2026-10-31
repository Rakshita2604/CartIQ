from collections.abc import Mapping

import pytest

from cartiq.config import BM25Settings, QuerySettings
from cartiq.domain import Product, Quantity, QuantityKind, TermSource, Unit
from cartiq.query import STAGES, QueryParser, Stages
from cartiq.query.lexicon import load_lexicon
from cartiq.retrieval import BM25FIndex


@pytest.fixture(scope="module")
def frequencies(synthetic_products: tuple[Product, ...]) -> dict[str, int]:
    index = BM25FIndex.build(synthetic_products, BM25Settings())
    return {term: index.document_frequency(term) for term in index.vocabulary}


def parser(frequencies: Mapping[str, int], stages: Stages) -> QueryParser:
    return QueryParser(
        frequencies, frequencies, load_lexicon("hinglish-v1"), QuerySettings(), stages
    )


def terms(concept) -> dict[str, tuple[str, float]]:  # type: ignore[no-untyped-def]
    return {t.term: (t.source.value, t.weight) for t in concept.terms}


def test_all_stages_off_gives_one_concept_per_unique_token(frequencies: dict[str, int]) -> None:
    parsed = parser(frequencies, Stages()).parse("Milk  amul milk 1 litre")

    assert [c.text for c in parsed.concepts] == ["1", "amul", "litre", "milk"]
    assert all(len(c.terms) == 1 for c in parsed.concepts)
    assert parsed.quantity_mention is None
    assert parsed.raw == "Milk  amul milk 1 litre"


def test_s1_extracts_quantity_and_keeps_spans(frequencies: dict[str, int]) -> None:
    parsed = parser(frequencies, STAGES["s1"]).parse("amul milk 1 litre")

    assert parsed.quantity == Quantity(1, Unit.L)
    assert parsed.quantity_mention is not None
    assert parsed.quantity_mention.spans == ((10, 17),)
    assert parsed.normalized == "amul milk"
    assert [c.text for c in parsed.concepts] == ["amul", "milk"]


def test_s1_does_not_expand_or_correct(frequencies: dict[str, int]) -> None:
    parsed = parser(frequencies, STAGES["s1"]).parse("doodh amull")

    assert all(len(c.terms) == 1 for c in parsed.concepts)
    assert parsed.corrections == ()


def test_s2_phrase_is_one_concept_with_split_weights(frequencies: dict[str, int]) -> None:
    (concept,) = parser(frequencies, STAGES["s2"]).parse("sarson ka tel").concepts

    assert concept.text == "sarson ka tel"
    assert terms(concept) == {
        "sarson": ("query", 0.5),
        "tel": ("query", 0.5),  # the connector "ka" is dropped
        "mustard": ("lexicon", 0.5),
        "oil": ("lexicon", 0.5),
    }


def test_s2_keeps_the_typed_word(frequencies: dict[str, int]) -> None:
    (concept,) = parser(frequencies, STAGES["s2"]).parse("doodh").concepts

    assert terms(concept) == {"doodh": ("query", 1.0), "milk": ("lexicon", 1.0)}


def test_broad_expansion_splits_weight_per_alternative(frequencies: dict[str, int]) -> None:
    (concept,) = parser(frequencies, STAGES["s2"]).parse("thanda").concepts

    assert terms(concept) == {
        "thanda": ("query", 1.0),
        "cold": ("lexicon", 0.5),
        "drink": ("lexicon", 0.5),
        "soft": ("lexicon", 0.5),
    }


def test_s3_correction_is_added_and_recorded(frequencies: dict[str, int]) -> None:
    parsed = parser(frequencies, STAGES["s3"]).parse("amull milk")

    amull = next(c for c in parsed.concepts if c.text == "amull")
    assert terms(amull) == {"amull": ("query", 1.0), "amul": ("spelling", 1.0)}
    assert [(c.original, c.corrected) for c in parsed.corrections] == [("amull", "amul")]


def test_s3_correction_to_a_lexicon_term_expands_it(frequencies: dict[str, int]) -> None:
    (concept,) = parser(frequencies, STAGES["s3"]).parse("dhanya").concepts

    # "dhania" and "dhaniya" are both one edit away and both lexicon spellings (df 0); the
    # alphabetical tie-break picks "dhania". Either expands to coriander.
    assert terms(concept) == {
        "dhanya": ("query", 1.0),
        "dhania": ("spelling", 1.0),
        "coriander": ("lexicon", 1.0),
    }


def test_s3_never_corrects_lexicon_words(frequencies: dict[str, int]) -> None:
    parsed = parser(frequencies, STAGES["s3"]).parse("gud kela")

    assert parsed.corrections == ()


def test_s4_adds_plural_forms_with_the_same_weight(frequencies: dict[str, int]) -> None:
    (concept,) = parser(frequencies, STAGES["s4"]).parse("masala").concepts

    assert terms(concept) == {"masala": ("query", 1.0), "masalas": ("inflection", 1.0)}


def test_quantity_only_query_has_no_concepts(frequencies: dict[str, int]) -> None:
    parsed = parser(frequencies, STAGES["s4"]).parse("12 pack")

    assert parsed.concepts == ()
    assert parsed.quantity_mention is not None
    assert parsed.quantity_mention.kind is QuantityKind.PACK


def test_typed_terms_are_always_kept(frequencies: dict[str, int]) -> None:
    for query in ("doodh", "amull", "sarson ka tel", "masala", "biscut", "thanda"):
        parsed = parser(frequencies, STAGES["s4"]).parse(query)
        typed = {t.term for c in parsed.concepts for t in c.terms if t.source is TermSource.QUERY}
        assert typed >= {w for w in query.split() if w != "ka"}


@pytest.fixture(scope="module")
def s5(synthetic_products: tuple[Product, ...]) -> QueryParser:
    from cartiq.query.factory import build_parser

    index = BM25FIndex.build(synthetic_products, BM25Settings())
    return build_parser(list(synthetic_products), index.vocabulary, QuerySettings(), STAGES["s5"])


class TestBrandStage:
    def test_alias_is_recognised_protected_and_weight_normalised(self, s5: QueryParser) -> None:
        parsed = s5.parse("coke 6 x 300 ml")

        assert (parsed.brand, parsed.brand_key) == ("Coca-Cola", "cocacola")
        assert parsed.corrections == ()
        (concept,) = parsed.concepts
        assert terms(concept) == {
            "coke": ("query", 1.0),
            "coca": ("brand", 0.5),
            "cola": ("brand", 0.5),
        }

    def test_brand_words_are_never_corrected(self, s5: QueryParser) -> None:
        parsed = s5.parse("coca cola")

        assert parsed.corrections == ()
        assert parsed.brand == "Coca-Cola"
        assert [c.text for c in parsed.concepts] == ["coca", "cola"]  # exact spelling: as before

    def test_misspelled_brand_is_recognised_after_correction(self, s5: QueryParser) -> None:
        for query, brand in (
            ("amull milk", "Amul"),
            ("colgat toothpaste", "Colgate"),
            ("magii noodles", "Maggi"),
        ):
            parsed = s5.parse(query)
            assert parsed.brand == brand, query
            assert len(parsed.corrections) == 1, query

    def test_possessive_variant_adds_the_catalog_spelling(self, s5: QueryParser) -> None:
        exact = s5.parse("haldiram bhujia")
        variant = s5.parse("haldirams bhujia")

        assert exact.brand == variant.brand == "Haldiram's"
        assert all(len(c.terms) == 1 for c in exact.concepts)  # no stray "s" term
        haldirams = next(c for c in variant.concepts if c.text == "haldirams")
        assert terms(haldirams) == {"haldirams": ("query", 1.0), "haldiram": ("brand", 1.0)}

    def test_guard_decisions_are_reported_with_their_evidence(self, s5: QueryParser) -> None:
        parsed = s5.parse("tata salt")

        assert parsed.brand == "Tata"  # "tata" is a brand word across all Tata brands
        assert parsed.brand_notes == (
            "'tata' used as brand 'Tata': 9/9 products containing 'tata' have it in a brand "
            "(100% >= 50%)",
        )

    def test_only_the_first_brand_is_used(self, s5: QueryParser) -> None:
        parsed = s5.parse("amul or nescafe")

        assert parsed.brand == "Amul"
        assert parsed.brand_notes[-1] == "also names brand 'Nescafe'; not used"

    def test_s4_sets_no_brand(self, frequencies: dict[str, int]) -> None:
        parsed = parser(frequencies, STAGES["s4"]).parse("coca cola")

        assert (parsed.brand, parsed.brand_key, parsed.brand_notes) == (None, None, ())


def test_brands_stage_needs_a_gazetteer(frequencies: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="BrandGazetteer"):
        parser(frequencies, STAGES["s5"])


@pytest.fixture(scope="module")
def bigbasket_like() -> QueryParser:
    from decimal import Decimal

    from cartiq.query.factory import build_parser

    def product(pid: str, name: str, brand: str, kind: str) -> Product:
        return Product(pid, f"{brand} {name}", ("Food", "Misc", kind), Decimal(10), brand=brand)

    products = [
        *(product(f"m{i}", "Masala Noodles", "MAGGI", "Instant Noodles") for i in range(3)),
        *(product(f"a{i}", "Face Wash", "Aroma Magic", "Face Care") for i in range(10)),
        *(product(f"x{i}", "Floor Mop", "Magix", "Mops") for i in range(5)),
        product("c1", "Soft Drink", "Coca-Cola", "Cold Drinks"),
        product("c2", "Cake", "Britannia", "Cakes"),
        product("h1", "Bhujia", "Haldirams", "Namkeen"),
        product("t1", "Fresh Tomato", "Farm", "Tomato"),
    ]
    index = BM25FIndex.build(products, BM25Settings())
    return build_parser(products, index.vocabulary, QuerySettings(), STAGES["s5"])


class TestPhoneticRegressions:
    """A BigBasket-like catalog where "magii" has four equally close brand-word targets."""

    def test_magii_resolves_to_the_sound_alike_brand(self, bigbasket_like: QueryParser) -> None:
        parsed = bigbasket_like.parse("magii noodles")

        assert [(c.original, c.corrected) for c in parsed.corrections] == [("magii", "maggi")]
        assert parsed.brand == "MAGGI"

    def test_coke_and_cola_are_protected(self, bigbasket_like: QueryParser) -> None:
        for query in ("coke", "coca cola", "cola"):
            assert bigbasket_like.parse(query).corrections == (), query
        assert bigbasket_like.parse("coke").brand == "Coca-Cola"

    def test_haldiram_variant_is_recognised_not_corrected(
        self, bigbasket_like: QueryParser
    ) -> None:
        parsed = bigbasket_like.parse("haldiram bhujia")

        assert parsed.corrections == ()
        assert parsed.brand == "Haldirams"

    def test_ordinary_words_get_no_brand(self, bigbasket_like: QueryParser) -> None:
        parsed = bigbasket_like.parse("tomatto")

        assert [(c.original, c.corrected) for c in parsed.corrections] == [("tomatto", "tomato")]
        assert parsed.brand is None


class TestBrandConcepts:
    def test_exact_alias_variant_and_corrected_brands(self, s5: QueryParser) -> None:
        assert s5.parse("amul butter").brand_concepts == ("amul",)
        assert s5.parse("mother dairy paneer").brand_concepts == ("mother", "dairy")
        assert s5.parse("coke 6 x 300 ml").brand_concepts == ("coke",)
        assert s5.parse("haldirams bhujia").brand_concepts == ("haldirams",)
        assert s5.parse("amull milk").brand_concepts == ("amull",)

    def test_no_brand_no_brand_concepts(self, s5: QueryParser) -> None:
        assert s5.parse("basmati rice").brand_concepts == ()
