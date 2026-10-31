"""Search-time value objects passed between query understanding, retrieval and ranking."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from cartiq.domain._validation import require_id
from cartiq.domain.catalog import Quantity


class Constraint(StrEnum):
    """Explicit user intent that may only be relaxed visibly, never silently."""

    BRAND = "brand"
    QUANTITY = "quantity"
    CATEGORY = "category"
    PRODUCT = "product"  # a matchable non-brand concept of the query is not matched


class MatchTier(StrEnum):
    """Which result section a product belongs to.

    EXACT satisfies every explicit constraint; RELAXED drops at least one (listed on the
    result); SUBSTITUTE stands in for an unavailable product.
    """

    EXACT = "exact"
    RELAXED = "relaxed"
    SUBSTITUTE = "substitute"


class SubstituteReason(StrEnum):
    SAME_CATEGORY_SIZE = "same_category_size"
    SAME_BRAND_OTHER_SIZE = "same_brand_other_size"
    SIMILAR_PRODUCT = "similar_product"
    CO_PURCHASE = "co_purchase"


class TermSource(StrEnum):
    """Where a concept term came from. Only QUERY terms were typed by the shopper."""

    QUERY = "query"
    LEXICON = "lexicon"
    SPELLING = "spelling"
    INFLECTION = "inflection"
    BRAND = "brand"  # catalog spelling of a recognised brand alias or variant


@dataclass(frozen=True, slots=True)
class ConceptTerm:
    term: str
    source: TermSource
    weight: float = 1.0  # share of the concept's term frequency this term carries

    def __post_init__(self) -> None:
        require_id("ConceptTerm.term", self.term)
        if not 0 < self.weight <= 1:
            raise ValueError("ConceptTerm.weight must be in (0, 1]")


Slot = frozenset[str]  # accepted forms of one word ("oil" or its plural "oils")
Alternative = tuple[Slot, ...]  # one phrase as ordered word slots


@dataclass(frozen=True, slots=True)
class QueryConcept:
    """One unit of meaning in a query.

    ``terms`` are what retrieval scores: their weighted term frequencies are summed and
    saturated once, so synonyms of one concept never count as independent matches.

    ``alternatives`` say what it takes to *contain* the concept (for exact matching): the
    phrases that express it, each as ordered word slots. A product contains the concept when
    every slot of at least one alternative is present ("sarson ka tel" = ("sarson", "tel")
    or ("mustard", "oil"|"oils")), so a multi-word phrase is never satisfied by one of its
    words. Empty means one single-word alternative per term.
    """

    text: str  # what the shopper typed for this concept, e.g. "sarson ka tel"
    terms: tuple[ConceptTerm, ...]
    alternatives: tuple[Alternative, ...] = ()

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("QueryConcept needs at least one term")
        if len({term.term for term in self.terms}) != len(self.terms):
            raise ValueError("QueryConcept terms must be unique")
        for alternative in self.alternatives:
            if not alternative or any(not slot for slot in alternative):
                raise ValueError("QueryConcept alternatives need non-empty word slots")

    @classmethod
    def single(cls, term: str) -> "QueryConcept":
        return cls(term, (ConceptTerm(term, TermSource.QUERY),), ((frozenset({term}),),))

    @property
    def phrases(self) -> tuple[Alternative, ...]:
        """``alternatives``, or one single-word alternative per term when none are given."""
        if self.alternatives:
            return self.alternatives
        return tuple((frozenset({term.term}),) for term in self.terms)


class CorrectionReason(StrEnum):
    UNKNOWN = "unknown"  # the typed token is not in the vocabulary
    DOMINANT = "dominant"  # known but rare; a near neighbour is far more frequent


@dataclass(frozen=True, slots=True)
class Correction:
    original: str
    corrected: str
    distance: int
    reason: CorrectionReason


class QuantityKind(StrEnum):
    AMOUNT = "amount"  # a size, optionally with a pack count: "500 ml", "6 x 300 ml"
    PACK = "pack"  # a pack count without a size: "12 pack", "pack of 6"
    BARE_NUMBER = "bare_number"  # a trailing integer: "eggs 12"; a count, ambiguous


@dataclass(frozen=True, slots=True)
class QuantityMention:
    """A quantity found in the query and the character spans of the query it came from.

    A mention can have several spans ("300 ml ... 6 pack"); they are sorted and disjoint.
    """

    kind: QuantityKind
    spans: tuple[tuple[int, int], ...]
    text: str
    quantity: Quantity | None = None
    pack_count: int | None = None
    count: int | None = None  # BARE_NUMBER only: N pieces or a pack of N, undecided

    def __post_init__(self) -> None:
        if not self.spans or any(not 0 <= start < end for start, end in self.spans):
            raise ValueError("QuantityMention needs non-empty spans")
        if any(a[1] > b[0] for a, b in zip(self.spans, self.spans[1:], strict=False)):
            raise ValueError("QuantityMention spans must be sorted and disjoint")
        expected = {
            QuantityKind.AMOUNT: self.quantity is not None and self.count is None,
            QuantityKind.PACK: self.quantity is None
            and self.pack_count is not None
            and self.count is None,
            QuantityKind.BARE_NUMBER: self.quantity is None
            and self.pack_count is None
            and self.count is not None,
        }
        if not expected[self.kind]:
            raise ValueError(f"QuantityMention fields do not match kind {self.kind}")


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    raw: str
    normalized: str  # the text used for retrieval (quantity spans removed)
    tokens: tuple[str, ...]
    concepts: tuple[QueryConcept, ...] = ()
    corrections: tuple[Correction, ...] = ()
    quantity_mention: QuantityMention | None = None
    brand: str | None = None  # display name of the recognised brand
    brand_key: str | None = None  # its match key: tokens joined ("cocacola")
    brand_notes: tuple[str, ...] = ()  # each brand decision with its evidence
    brand_concepts: tuple[str, ...] = ()  # texts of the concepts that name the brand
    category: str | None = None
    quantity: Quantity | None = None
    pack_count: int | None = None

    @property
    def constraints(self) -> frozenset[Constraint]:
        present = set()
        if self.brand is not None:
            present.add(Constraint.BRAND)
        if self.quantity is not None or self.pack_count is not None:
            present.add(Constraint.QUANTITY)
        if self.category is not None:
            present.add(Constraint.CATEGORY)
        return frozenset(present)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieved product with its per-source relevance signals. No availability here."""

    product_id: str
    fused_score: float
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None

    def __post_init__(self) -> None:
        require_id("Candidate.product_id", self.product_id)
        if not math.isfinite(self.fused_score):
            raise ValueError("Candidate.fused_score must be finite")
        for name, rank in (("bm25_rank", self.bm25_rank), ("dense_rank", self.dense_rank)):
            if rank is not None and rank < 1:
                raise ValueError(f"Candidate.{name} must be >= 1 (ranks are 1-based)")


@dataclass(frozen=True, slots=True)
class ScoredProduct:
    product_id: str
    score: float
    tier: MatchTier
    relaxed: frozenset[Constraint] = frozenset()
    features: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id("ScoredProduct.product_id", self.product_id)
        if self.tier is MatchTier.RELAXED and not self.relaxed:
            raise ValueError("RELAXED results must state which constraints were relaxed")
        if self.tier is MatchTier.EXACT and self.relaxed:
            raise ValueError("EXACT results cannot have relaxed constraints")


@dataclass(frozen=True, slots=True)
class Substitute:
    original_id: str
    substitute_id: str
    score: float
    reason: SubstituteReason

    def __post_init__(self) -> None:
        require_id("Substitute.original_id", self.original_id)
        require_id("Substitute.substitute_id", self.substitute_id)
        if self.original_id == self.substitute_id:
            raise ValueError("A product cannot substitute for itself")
