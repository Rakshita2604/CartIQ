"""Hand-calculated metric fixtures. Expected values are worked out in the comments, not by
the code under test. log2(2) = 1, log2(3) = 1.58496..., log2(4) = 2; gain = 2^grade - 1, so
grades 3/2/1/0 give gains 7/3/1/0."""

from collections.abc import Callable

import pytest

from cartiq.domain import Grade
from cartiq.evaluation import judged_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

E, R, L, Z = Grade.EXACT, Grade.RELEVANT, Grade.RELATED, Grade.IRRELEVANT  # Z: zero
U = None  # unjudged


class TestNdcg:
    def test_ideal_ranking(self) -> None:
        assert ndcg_at_k([E, R, L], [E, R, L], k=10) == pytest.approx(1.0)

    def test_mixed_ranking_with_unjudged(self) -> None:
        # DCG  = 1/log2(2) + 0 + 7/log2(4)          = 1 + 3.5             = 4.5
        # IDCG = 7/log2(2) + 3/log2(3) + 1/log2(4)  = 7 + 1.89279 + 0.5   = 9.39279
        assert ndcg_at_k([L, U, E], [E, R, L], k=10) == pytest.approx(0.47909091485969846)

    def test_missing_relevant_product_lowers_score(self) -> None:
        # DCG = 7; IDCG = 7 + 7/log2(3) = 11.41650
        assert ndcg_at_k([E], [E, E], k=10) == pytest.approx(0.6131471927654584)

    def test_only_top_k_counts_for_ranking_and_ideal(self) -> None:
        assert ndcg_at_k([Z, E], [E], k=1) == 0.0
        # Ideal truncated to its top 1 (7): ranking [E] is ideal at k=1.
        assert ndcg_at_k([E], [E, E, R], k=1) == pytest.approx(1.0)

    def test_empty_retrieval_with_relevant_products_scores_zero(self) -> None:
        assert ndcg_at_k([], [R], k=10) == 0.0

    def test_no_ideal_gain_is_undefined(self) -> None:
        assert ndcg_at_k([E], [], k=10) is None
        assert ndcg_at_k([], [Z, Z], k=10) is None

    def test_fewer_results_than_k(self) -> None:
        # DCG = 3; IDCG = 3 (one judged relevant product)
        assert ndcg_at_k([R], [R], k=10) == pytest.approx(1.0)


class TestReciprocalRank:
    RANKED = (U, L, R, E)

    @pytest.mark.parametrize(("min_grade", "expected"), [(R, 1 / 3), (E, 1 / 4), (L, 1 / 2)])
    def test_uses_the_configured_min_grade(self, min_grade: Grade, expected: float) -> None:
        assert reciprocal_rank(self.RANKED, 10, min_grade, possible_hits=2) == pytest.approx(
            expected
        )

    def test_unjudged_is_never_a_hit(self) -> None:
        assert reciprocal_rank([U, U], 10, L, possible_hits=1) == 0.0

    def test_hit_beyond_k_scores_zero(self) -> None:
        assert reciprocal_rank([Z, Z, E], 2, R, possible_hits=1) == 0.0

    def test_empty_retrieval_scores_zero(self) -> None:
        assert reciprocal_rank([], 10, R, possible_hits=3) == 0.0

    def test_no_possible_hit_is_undefined(self) -> None:
        assert reciprocal_rank([L, Z], 10, R, possible_hits=0) is None


class TestRecall:
    RANKED = (E, Z, R, U, R)

    @pytest.mark.parametrize(("k", "expected"), [(1, 1 / 4), (3, 2 / 4), (5, 3 / 4), (100, 3 / 4)])
    def test_hits_over_relevant(self, k: int, expected: float) -> None:
        assert recall_at_k(self.RANKED, k, R, relevant=4) == pytest.approx(expected)

    def test_min_grade_exact(self) -> None:
        assert recall_at_k(self.RANKED, 5, E, relevant=1) == pytest.approx(1.0)

    def test_empty_retrieval_scores_zero(self) -> None:
        assert recall_at_k([], 10, R, relevant=2) == 0.0

    def test_no_relevant_products_is_undefined(self) -> None:
        assert recall_at_k([L, Z], 10, R, relevant=0) is None

    def test_more_hits_than_relevant_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="exceed"):
            recall_at_k([E, E], 10, R, relevant=1)


class TestJudged:
    def test_share_of_judged_results(self) -> None:
        assert judged_at_k([E, U, Z, U], 10) == pytest.approx(2 / 4)
        assert judged_at_k([E, U, Z, U], 1) == pytest.approx(1.0)

    def test_empty_results_are_undefined(self) -> None:
        assert judged_at_k([], 10) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: ndcg_at_k([E], [E], k=0),
        lambda: reciprocal_rank([E], 0, R, possible_hits=1),
        lambda: recall_at_k([E], 0, R, relevant=1),
        lambda: judged_at_k([E], 0),
    ],
)
def test_k_must_be_positive(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="k must be"):
        call()
