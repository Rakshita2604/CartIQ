from typing import Any

import pytest

from cartiq.domain import (
    Coverage,
    Grade,
    Interpretation,
    InterpretationShare,
    JudgedQuery,
    QueryAttribute,
    QuerySegment,
)

DOMINANT = InterpretationShare.DOMINANT
MINORITY = InterpretationShare.MINORITY
FRINGE = InterpretationShare.FRINGE


def make(**overrides: Any) -> JudgedQuery:
    fields: dict[str, Any] = {
        "query_id": "q1",
        "text": "doodh",
        "segment": QuerySegment.HINGLISH,
        "grades": {"P1": Grade.EXACT, "P2": Grade.RELATED, "P3": Grade.IRRELEVANT},
        "labeler": "manual:v1",
    }
    fields.update(overrides)
    return JudgedQuery(**fields)


def oil(**overrides: Any) -> JudgedQuery:
    """An ambiguous query: cooking oil (dominant) versus hair oil (minority)."""
    fields: dict[str, Any] = {
        "text": "oil",
        "segment": QuerySegment.AMBIGUOUS,
        "grades": {"cook": Grade.EXACT, "hair": Grade.RELEVANT},
        "interpretations": (
            Interpretation("cooking oil", DOMINANT, frozenset({"cook"})),
            Interpretation("hair oil", MINORITY, frozenset({"hair"})),
        ),
    }
    fields.update(overrides)
    return make(**fields)


class TestCoverage:
    def test_partial_leaves_unlisted_products_unjudged(self) -> None:
        query = make()

        assert query.grade("P1") is Grade.EXACT
        assert query.grade("P3") is Grade.IRRELEVANT  # explicitly judged 0
        assert query.grade("P9") is None  # unjudged, not a confirmed negative

    def test_exhaustive_judges_unlisted_products_irrelevant(self) -> None:
        assert make(coverage=Coverage.EXHAUSTIVE).grade("P9") is Grade.IRRELEVANT


def test_collections_are_read_only_copies() -> None:
    grades = {"P1": Grade.EXACT}
    query = make(grades=grades)
    grades["P2"] = Grade.EXACT

    assert "P2" not in query.grades
    with pytest.raises(TypeError):
        query.grades["P3"] = Grade.EXACT  # type: ignore[index]


def test_segment_implies_its_attribute() -> None:
    query = make(attributes=frozenset({QueryAttribute.BRAND}))

    assert query.attributes == {QueryAttribute.BRAND, QueryAttribute.TRANSLITERATION}
    assert make(segment=QuerySegment.HEAD).attributes == frozenset()


def test_products_at_least() -> None:
    assert make().products_at_least(Grade.RELATED) == {"P1", "P2"}


class TestNoExactMatchFlag:
    def test_query_without_exact_product_must_be_flagged(self) -> None:
        with pytest.raises(ValueError, match="no_exact_match"):
            make(grades={"P2": Grade.RELEVANT})

    def test_flagged_query_may_have_no_relevant_product_at_all(self) -> None:
        query = make(grades={}, attributes=frozenset({QueryAttribute.NO_EXACT_MATCH}))

        assert query.products_at_least(Grade.RELATED) == frozenset()

    def test_flag_contradicting_an_exact_product_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has EXACT"):
            make(attributes=frozenset({QueryAttribute.NO_EXACT_MATCH}))


class TestReasons:
    def test_reasons_for_grades_zero_to_two_are_kept(self) -> None:
        query = make(reasons={"P2": "neighbouring type", "P3": "lexical trap"})

        assert query.reasons["P3"] == "lexical trap"

    @pytest.mark.parametrize(
        ("reasons", "message"),
        [
            ({"P1": "it is milk"}, "EXACT"),
            ({"P9": "unlisted"}, "ungraded"),
            ({"P2": " "}, "blank reason"),
        ],
    )
    def test_invalid_reasons(self, reasons: dict[str, str], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            make(reasons=reasons)


class TestInterpretations:
    def test_valid_interpretation_set(self) -> None:
        assert len(oil().interpretations) == 2

    def test_interpretation_may_have_no_products(self) -> None:
        query = oil(
            grades={"masala": Grade.RELEVANT},
            attributes=frozenset({QueryAttribute.NO_EXACT_MATCH}),
            interpretations=(
                Interpretation("chicken meat", DOMINANT),
                Interpretation("chicken masala", MINORITY, frozenset({"masala"})),
            ),
        )

        assert query.interpretations[0].products == frozenset()

    def test_several_dominant_interpretations_may_each_reach_exact(self) -> None:
        oil(
            grades={"cook": Grade.EXACT, "hair": Grade.EXACT},
            interpretations=(
                Interpretation("cooking oil", DOMINANT, frozenset({"cook"})),
                Interpretation("hair oil", DOMINANT, frozenset({"hair"})),
            ),
        )

    def test_product_in_several_interpretations_takes_the_highest_cap(self) -> None:
        oil(
            grades={"cook": Grade.EXACT, "both": Grade.EXACT},
            interpretations=(
                Interpretation("cooking oil", DOMINANT, frozenset({"cook", "both"})),
                Interpretation("hair oil", MINORITY, frozenset({"both"})),
            ),
        )

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            (
                {"grades": {"cook": Grade.EXACT, "hair": Grade.EXACT}},
                "above its interpretation cap",
            ),
            (
                {
                    "grades": {"cook": Grade.EXACT, "hair": Grade.RELEVANT, "x": Grade.RELATED},
                },
                "x belongs to no interpretation",
            ),
            (
                {
                    "interpretations": (
                        Interpretation("cooking oil", DOMINANT, frozenset({"cook", "nope"})),
                        Interpretation("hair oil", MINORITY, frozenset({"hair"})),
                    )
                },
                "nope in 'cooking oil' is not graded",
            ),
            (
                {
                    "interpretations": (
                        Interpretation("cooking oil", MINORITY, frozenset({"cook"})),
                        Interpretation("hair oil", FRINGE, frozenset({"hair"})),
                    )
                },
                "no dominant interpretation",
            ),
            (
                {
                    "interpretations": (
                        Interpretation("cooking oil", DOMINANT, frozenset({"cook"})),
                    )
                },
                "at least two",
            ),
            (
                {
                    "interpretations": (
                        Interpretation("oil", DOMINANT, frozenset({"cook"})),
                        Interpretation("oil", MINORITY, frozenset({"hair"})),
                    )
                },
                "duplicate interpretation",
            ),
            ({"segment": QuerySegment.HEAD}, "require multi_intent"),
        ],
    )
    def test_invalid_interpretations(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            oil(**overrides)


def test_requires_grade_enum() -> None:
    with pytest.raises(ValueError, match="Grade"):
        make(grades={"P1": 3})


def test_requires_labeler() -> None:
    with pytest.raises(ValueError, match="labeler"):
        make(labeler="")
