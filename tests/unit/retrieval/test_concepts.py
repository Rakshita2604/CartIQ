"""Concept scoring (BM25F with weighted synonym folding) against hand calculations.

Documents (name only, w = 1, b = 0.75, avglen 2, k1 = 1.2):
    d1 "a b" (len 2): B = 1        -> tf~(a) = 1
    d2 "a a c" (len 3): B = 1.375  -> tf~(a) = 2/1.375 = 1.454545, tf~(c) = 0.727273
    d3 "c" (len 1):   B = 0.625    -> tf~(c) = 1.6
"""

import math

import pytest

from cartiq.config import BM25Settings, FieldWeight
from cartiq.domain import ConceptTerm, QueryConcept, TermSource
from cartiq.retrieval import BM25FIndex, Field

OFF = FieldWeight(weight=0, b=0)
NAME_ONLY = BM25Settings(
    k1=1.2, **{f.value: (FieldWeight(weight=1, b=0.75) if f is Field.NAME else OFF) for f in Field}
)
DOCS = {
    "d1": {Field.NAME: ["a", "b"]},
    "d2": {Field.NAME: ["a", "a", "c"]},
    "d3": {Field.NAME: ["c"]},
}


def index() -> BM25FIndex:
    return BM25FIndex.from_documents(list(DOCS), list(DOCS.values()), NAME_ONLY)


def concept(*terms: tuple[str, float]) -> QueryConcept:
    return QueryConcept(
        "c",
        tuple(
            ConceptTerm(t, TermSource.QUERY if i == 0 else TermSource.LEXICON, w)
            for i, (t, w) in enumerate(terms)
        ),
    )


def scores(concepts: list[QueryConcept]) -> dict[str, float]:
    return {c.product_id: c.bm25_score or 0.0 for c in index().search_concepts(concepts, 10)}


def bm25(idf: float, tf: float) -> float:
    return idf * tf / (1.2 + tf)


def test_single_term_concepts_equal_plain_search() -> None:
    plain = index().search("a c", 10)

    assert (
        index().search_concepts([QueryConcept.single("a"), QueryConcept.single("c")], 10) == plain
    )


def test_union_df_and_one_saturation() -> None:
    result = scores([concept(("a", 1.0), ("c", 1.0))])
    idf = math.log(1 + (3 - 3 + 0.5) / (3 + 0.5))  # df_C = 3: every document matches a or c

    assert result["d1"] == pytest.approx(bm25(idf, 1.0))
    # d2 matches both alternatives: tf~ is summed, then saturated once.
    assert result["d2"] == pytest.approx(bm25(idf, 2 / 1.375 + 1 / 1.375))
    assert result["d2"] < idf
    assert result["d3"] == pytest.approx(bm25(idf, 1.6))


def test_matching_several_alternatives_scores_less_than_separate_terms() -> None:
    folded = scores([concept(("a", 1.0), ("c", 1.0))])["d2"]
    separate = scores([QueryConcept.single("a"), QueryConcept.single("c")])["d2"]

    assert folded < separate


def test_term_weights_scale_term_frequency() -> None:
    result = scores([concept(("a", 1.0), ("c", 0.5))])
    idf = math.log(1 + 0.5 / 3.5)

    assert result["d3"] == pytest.approx(bm25(idf, 0.5 * 1.6))
    assert result["d2"] == pytest.approx(bm25(idf, 2 / 1.375 + 0.5 * (1 / 1.375)))


def test_broad_expansion_gives_weak_matches_less_credit() -> None:
    docs = {
        "cola": {Field.NAME: ["soft", "drink"], Field.PRODUCT_TYPE: ["cold", "drinks"]},
        "mango": {Field.NAME: ["mango", "drink"]},
        "other": {Field.NAME: ["milk"]},
    }
    settings = BM25Settings(
        k1=1.2,
        **{
            f.value: FieldWeight(weight={Field.NAME: 3, Field.PRODUCT_TYPE: 2}.get(f, 0), b=0)
            for f in Field
        },
    )
    idx = BM25FIndex.from_documents(list(docs), list(docs.values()), settings)
    thanda = QueryConcept(
        "thanda",
        (
            ConceptTerm("thanda", TermSource.QUERY),
            *(ConceptTerm(t, TermSource.LEXICON, 0.5) for t in ("cold", "drink", "soft", "drinks")),
        ),
    )

    result = {c.product_id: c.bm25_score or 0.0 for c in idx.search_concepts([thanda], 10)}
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))

    # cola: 0.5 * (3 soft + 3 drink + 2 cold + 2 drinks) = 5; mango: 0.5 * 3 = 1.5
    assert result["cola"] == pytest.approx(bm25(idf, 5.0))
    assert result["mango"] == pytest.approx(bm25(idf, 1.5))
    assert result["mango"] < result["cola"] < idf


def test_explain_reports_each_concept_and_sums_to_the_score() -> None:
    concepts = [
        concept(("a", 1.0), ("c", 1.0)),
        QueryConcept.single("b"),
        QueryConcept.single("zzz"),
    ]
    score = scores(concepts)["d1"]

    explained = index().explain(concepts, "d1")

    assert sum(e.contribution for e in explained) == pytest.approx(score)
    first, second, missing = explained
    assert (first.document_frequency, first.weighted_tf) == (3, 1.0)
    assert [(t.term, t.weighted_tf) for t in first.terms] == [("a", 1.0), ("c", 0.0)]
    assert second.contribution == pytest.approx(bm25(math.log(1 + 2.5 / 1.5), 1.0))
    assert (missing.document_frequency, missing.contribution) == (0, 0.0)


def test_concepts_matching_nothing_return_nothing() -> None:
    assert index().search_concepts([QueryConcept.single("zzz")], 10) == []
    assert index().search_concepts([], 10) == []


def test_match_count() -> None:
    concepts = [QueryConcept.single("a"), QueryConcept.single("zzz")]

    assert index().match_count(concepts) == 2
    assert index().match_count(concepts, allowed={"d1"}) == 1
