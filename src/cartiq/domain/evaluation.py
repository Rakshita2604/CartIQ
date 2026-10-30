"""Offline relevance judgments.

Judgments are graded against the catalog by a labeler, never derived from the output of
the ranker under evaluation, so they record who or what produced them.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from types import MappingProxyType

from cartiq.domain._validation import require_id


class QuerySegment(StrEnum):
    HEAD = "head"
    MISSPELLED = "misspelled"
    HINGLISH = "hinglish"
    BRAND = "brand"
    QUANTITY = "quantity"
    AMBIGUOUS = "ambiguous"


class QueryAttribute(StrEnum):
    """Secondary traits of a query; unlike segments, a query can have several."""

    BRAND = "brand"
    QUANTITY = "quantity"
    TRANSLITERATION = "transliteration"
    MISSPELLING = "misspelling"
    MULTI_INTENT = "multi_intent"
    NO_EXACT_MATCH = "no_exact_match"


# Each primary segment implies one attribute, added automatically.
SEGMENT_ATTRIBUTE: Mapping[QuerySegment, QueryAttribute] = MappingProxyType(
    {
        QuerySegment.MISSPELLED: QueryAttribute.MISSPELLING,
        QuerySegment.HINGLISH: QueryAttribute.TRANSLITERATION,
        QuerySegment.BRAND: QueryAttribute.BRAND,
        QuerySegment.QUANTITY: QueryAttribute.QUANTITY,
        QuerySegment.AMBIGUOUS: QueryAttribute.MULTI_INTENT,
    }
)


class Grade(IntEnum):
    IRRELEVANT = 0
    RELATED = 1
    RELEVANT = 2
    EXACT = 3


class Coverage(StrEnum):
    """EXHAUSTIVE: every catalog product was considered, so unlisted means grade 0.
    PARTIAL: unlisted products are unjudged."""

    EXHAUSTIVE = "exhaustive"
    PARTIAL = "partial"


class InterpretationShare(StrEnum):
    DOMINANT = "dominant"
    MINORITY = "minority"
    FRINGE = "fringe"

    @property
    def max_grade(self) -> Grade:
        return _SHARE_MAX_GRADE[self]


_SHARE_MAX_GRADE = {
    InterpretationShare.DOMINANT: Grade.EXACT,
    InterpretationShare.MINORITY: Grade.RELEVANT,
    InterpretationShare.FRINGE: Grade.RELATED,
}


@dataclass(frozen=True, slots=True)
class Interpretation:
    """One plausible meaning of an ambiguous query and the products that serve it.

    ``products`` may be empty: the catalog then has nothing for this meaning.
    """

    name: str
    share: InterpretationShare
    products: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Interpretation.name must not be blank")


@dataclass(frozen=True, slots=True)
class JudgedQuery:
    query_id: str
    text: str
    segment: QuerySegment
    grades: Mapping[str, Grade]
    labeler: str
    coverage: Coverage = Coverage.PARTIAL
    attributes: frozenset[QueryAttribute] = frozenset()
    interpretations: tuple[Interpretation, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id("JudgedQuery.query_id", self.query_id)
        require_id("JudgedQuery.labeler", self.labeler)
        if not self.text.strip():
            raise ValueError("JudgedQuery.text must not be blank")
        for product_id, grade in self.grades.items():
            require_id("JudgedQuery product id", product_id)
            if not isinstance(grade, Grade):
                raise ValueError(f"grade for {product_id} must be a Grade")

        attributes = set(self.attributes)
        if self.segment in SEGMENT_ATTRIBUTE:
            attributes.add(SEGMENT_ATTRIBUTE[self.segment])
        # Freeze collections so a shared judgment set cannot be mutated by a caller.
        object.__setattr__(self, "attributes", frozenset(attributes))
        object.__setattr__(self, "grades", MappingProxyType(dict(self.grades)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))

        self._check_exact_match_flag()
        self._check_reasons()
        self._check_interpretations()

    def grade(self, product_id: str) -> Grade | None:
        """The product's grade; None when it is unjudged (only possible with PARTIAL)."""
        grade = self.grades.get(product_id)
        if grade is None and self.coverage is Coverage.EXHAUSTIVE:
            return Grade.IRRELEVANT
        return grade

    def products_at_least(self, grade: Grade) -> frozenset[str]:
        return frozenset(pid for pid, value in self.grades.items() if value >= grade)

    def _check_exact_match_flag(self) -> None:
        # Guards against an accidentally empty label: a query without an exact product must
        # say so explicitly.
        has_exact = any(grade is Grade.EXACT for grade in self.grades.values())
        flagged = QueryAttribute.NO_EXACT_MATCH in self.attributes
        if has_exact and flagged:
            raise ValueError(f"{self.query_id}: no_exact_match but has EXACT products")
        if not has_exact and not flagged:
            raise ValueError(f"{self.query_id}: no EXACT product; add the no_exact_match attribute")

    def _check_reasons(self) -> None:
        for product_id, reason in self.reasons.items():
            if not reason.strip():
                raise ValueError(f"{self.query_id}: blank reason for {product_id}")
            grade = self.grades.get(product_id)
            if grade is None:
                raise ValueError(f"{self.query_id}: reason for ungraded product {product_id}")
            if grade is Grade.EXACT:
                raise ValueError(f"{self.query_id}: EXACT product {product_id} takes no reason")

    def _check_interpretations(self) -> None:
        interpretations = self.interpretations
        if QueryAttribute.MULTI_INTENT not in self.attributes:
            if interpretations:
                raise ValueError(f"{self.query_id}: interpretations require multi_intent")
            return
        if len(interpretations) < 2:
            raise ValueError(f"{self.query_id}: multi_intent needs at least two interpretations")
        _require_unique(self.query_id, (i.name for i in interpretations))
        if not any(i.share is InterpretationShare.DOMINANT for i in interpretations):
            raise ValueError(f"{self.query_id}: no dominant interpretation")

        cap: dict[str, Grade] = {}
        for interpretation in interpretations:
            for product_id in interpretation.products:
                if self.grades.get(product_id, Grade.IRRELEVANT) < Grade.RELATED:
                    raise ValueError(
                        f"{self.query_id}: {product_id} in {interpretation.name!r} is not graded"
                        " 1 or higher"
                    )
                cap[product_id] = max(
                    cap.get(product_id, Grade.IRRELEVANT), interpretation.share.max_grade
                )
        for product_id in self.products_at_least(Grade.RELATED):
            if product_id not in cap:
                raise ValueError(f"{self.query_id}: {product_id} belongs to no interpretation")
            if self.grades[product_id] > cap[product_id]:
                raise ValueError(
                    f"{self.query_id}: {product_id} graded {self.grades[product_id].name} above"
                    f" its interpretation cap {cap[product_id].name}"
                )


def _require_unique(query_id: str, names: Iterable[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"{query_id}: duplicate interpretation {name!r}")
        seen.add(name)
