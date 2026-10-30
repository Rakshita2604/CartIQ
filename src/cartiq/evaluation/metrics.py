"""Ranking metrics over the grades of a ranked list.

Inputs are grades in rank order, with ``None`` for an unjudged product. Unjudged products
contribute zero gain; when a query's judgments are partial, the result is an estimate
(the ideal ranking may be missing unjudged relevant products too), not a bound.

A metric returns ``None`` when it is undefined because its denominator is zero: no ideal
gain (NDCG), no possible hit (MRR, Recall) or no results to inspect (Judged@k). Undefined
values are excluded from averages, never scored as 0 or 1. An empty ranking with a non-zero
denominator scores 0.
"""

import math
from collections.abc import Collection, Sequence

from cartiq.domain import Grade


def ndcg_at_k(ranked: Sequence[Grade | None], ideal: Collection[Grade], k: int) -> float | None:
    """DCG of the top ``k`` over DCG of the best top ``k`` from ``ideal`` (gain 2^g - 1)."""
    _check_k(k)
    ideal_dcg = _dcg(sorted(ideal, reverse=True)[:k])
    if ideal_dcg == 0:
        return None
    return _dcg([Grade.IRRELEVANT if grade is None else grade for grade in ranked[:k]]) / ideal_dcg


def reciprocal_rank(
    ranked: Sequence[Grade | None], k: int, min_grade: Grade, possible_hits: int
) -> float | None:
    """1 / rank of the first result graded at least ``min_grade`` within the top ``k``.

    ``possible_hits`` is how many such products could have been returned; zero makes the
    metric undefined.
    """
    _check_k(k)
    if possible_hits == 0:
        return None
    for rank, grade in enumerate(ranked[:k], start=1):
        if grade is not None and grade >= min_grade:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranked: Sequence[Grade | None], k: int, min_grade: Grade, relevant: int
) -> float | None:
    """Share of the ``relevant`` products (graded at least ``min_grade``) in the top ``k``."""
    _check_k(k)
    if relevant == 0:
        return None
    hits = sum(1 for grade in ranked[:k] if grade is not None and grade >= min_grade)
    if hits > relevant:
        raise ValueError(f"{hits} hits exceed the {relevant} relevant products")
    return hits / relevant


def judged_at_k(ranked: Sequence[Grade | None], k: int) -> float | None:
    """Share of the top ``k`` results that carry a judgment."""
    _check_k(k)
    top = ranked[:k]
    if not top:
        return None
    return sum(1 for grade in top if grade is not None) / len(top)


def _dcg(grades: Sequence[Grade]) -> float:
    return sum(((1 << grade) - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
