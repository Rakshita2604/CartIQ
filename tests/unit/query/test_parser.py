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
    return QueryParser(frequencies, load_lexicon("hinglish-v1"), QuerySettings(), stages)


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
