"""Offline relevance judgments.

Judgments are graded against the catalog by a labeler, never derived from the output of
the ranker under evaluation, so they record who or what produced them.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from cartiq.domain._validation import require_id


class QuerySegment(StrEnum):
    HEAD = "head"
    MISSPELLED = "misspelled"
    HINGLISH = "hinglish"
    BRAND = "brand"
    QUANTITY = "quantity"
    AMBIGUOUS = "ambiguous"


class Grade(IntEnum):
    IRRELEVANT = 0
    RELATED = 1
    RELEVANT = 2
    EXACT = 3


@dataclass(frozen=True, slots=True)
class Judgment:
    query_id: str
    query: str
    segment: QuerySegment
    product_id: str
    grade: Grade
    labeler: str

    def __post_init__(self) -> None:
        require_id("Judgment.query_id", self.query_id)
        require_id("Judgment.product_id", self.product_id)
        require_id("Judgment.labeler", self.labeler)
        if not self.query.strip():
            raise ValueError("Judgment.query must not be blank")
        if not isinstance(self.grade, Grade):
            raise ValueError("Judgment.grade must be a Grade")
