"""Constraint evaluation and constraint-first ordering (stage s6).

A candidate is EXACT when it satisfies every constraint the query states, RELAXED otherwise,
and a RELAXED result lists exactly the constraints it violates:

- BRAND: the query names a brand; the product's brand key must equal it. A product without
  a brand violates it.
- QUANTITY: the query states a quantity (``QuantityMention``):
  - a size (AMOUNT): the product's per-unit size must be the same amount in the same
    dimension (1 l = 1000 ml; 500 g never equals 500 ml). Any pack count is accepted unless
    the query states one, which must then equal the product's. The per-unit size is always
    compared, never the pack total (6 x 200 ml is not 1200 ml here).
  - a pack count only (PACK): the product's pack count must equal it.
  - a bare count N (BARE_NUMBER): the product is N pieces, or a pack of N.
  An unknown product size never satisfies a size or a bare count.
- PRODUCT: every matchable non-brand concept of the query must be matched by the product
  (any of the concept's alternative terms counts). Concepts no catalog product matches, and
  concepts that name the brand, are not required.

Ordering keeps retrieval's scores and ranks untouched: EXACT candidates first, then RELAXED,
each group in the original BM25 order (score descending, then product id). No other
heuristic is applied.
"""

from collections.abc import Collection, Mapping, Sequence

from cartiq.domain import (
    Candidate,
    Constraint,
    MatchTier,
    ParsedQuery,
    Product,
    Quantity,
    QuantityKind,
    QuantityMention,
    ScoredProduct,
    Unit,
)
from cartiq.query.brands import brand_key


def quantity_satisfied(mention: QuantityMention, product: Product) -> bool:
    if mention.kind is QuantityKind.AMOUNT:
        if mention.quantity is None or product.quantity is None:
            return False
        if not mention.quantity.same_amount(product.quantity):
            return False
        return mention.pack_count is None or product.pack_count == mention.pack_count
    if mention.kind is QuantityKind.PACK:
        return product.pack_count == mention.pack_count
    count = mention.count
    if count is None:
        return False
    pieces = product.quantity is not None and product.quantity.same_amount(
        Quantity(count, Unit.PCS)
    )
    return pieces or product.pack_count == count


def violations(
    parsed: ParsedQuery,
    product: Product,
    covered: Collection[str],
    required: Collection[str],
) -> frozenset[Constraint]:
    """Constraints ``product`` violates. ``covered`` are the concept texts it matches;
    ``required`` the concept texts it must match (matchable, non-brand)."""
    found = set()
    if parsed.brand_key is not None and (
        product.brand is None or brand_key(product.brand) != parsed.brand_key
    ):
        found.add(Constraint.BRAND)
    mention = parsed.quantity_mention
    if mention is not None and not quantity_satisfied(mention, product):
        found.add(Constraint.QUANTITY)
    if not set(required) <= set(covered):
        found.add(Constraint.PRODUCT)
    return frozenset(found)


def order(
    candidates: Sequence[Candidate],
    products: Mapping[str, Product],
    parsed: ParsedQuery,
    coverage: Mapping[str, Collection[str]],
    required: Collection[str],
) -> list[ScoredProduct]:
    """EXACT then RELAXED, each in the candidates' given (BM25) order."""
    exact: list[ScoredProduct] = []
    relaxed: list[ScoredProduct] = []
    for candidate in candidates:
        broken = violations(
            parsed, products[candidate.product_id], coverage[candidate.product_id], required
        )
        scored = ScoredProduct(
            product_id=candidate.product_id,
            score=candidate.bm25_score if candidate.bm25_score is not None else 0.0,
            tier=MatchTier.RELAXED if broken else MatchTier.EXACT,
            relaxed=broken,
            features={"bm25_rank": float(candidate.bm25_rank or 0)},
        )
        (relaxed if broken else exact).append(scored)
    return exact + relaxed
