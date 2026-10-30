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


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    corrected_tokens: tuple[str, ...] = ()
    expansions: tuple[str, ...] = ()
    brand: str | None = None
    category: str | None = None
    quantity: Quantity | None = None
    product_terms: tuple[str, ...] = ()

    @property
    def constraints(self) -> frozenset[Constraint]:
        present = set()
        if self.brand is not None:
            present.add(Constraint.BRAND)
        if self.quantity is not None:
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
