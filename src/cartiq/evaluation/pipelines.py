"""Search pipelines that can be evaluated. Each exposes whole-catalog retrieval (ignoring
stock) and per-store staged results. Neither has fallbacks or substitutions yet, so final
results are the in-stock matches."""

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from cartiq.config import BM25Settings, QuerySettings, RankingSettings
from cartiq.domain import Candidate, MatchTier, ParsedQuery, Product, ScoredProduct
from cartiq.evaluation.runner import StagedResults
from cartiq.parsing.text import tokenize
from cartiq.query import Stages
from cartiq.query.factory import build_parser, parser_identity
from cartiq.ranking import CoverageIndex, order
from cartiq.retrieval import BM25FIndex, bm25_config_id


class TokenOverlapPipeline:
    """A deliberately naive reference: count of query tokens found in the product text.

    It exists to exercise the evaluation harness end to end and to give later phases a floor
    to beat. It has no term weighting, spelling correction, synonyms, Hinglish or quantity
    understanding.
    """

    name = "token-overlap"

    def __init__(self, products: Iterable[Product], in_stock: Mapping[str, frozenset[str]]) -> None:
        self._documents = [
            (product.product_id, frozenset(tokenize(_overlap_document(product))))
            for product in sorted(products, key=lambda product: product.product_id)
        ]
        self._in_stock = in_stock

    def retrieve(self, query: str, depth: int) -> Sequence[str]:
        return self._ranked(query)[:depth]

    def search(self, query: str, store_id: str, limit: int) -> StagedResults:
        stocked = self._in_stock.get(store_id, frozenset())
        available = [product_id for product_id in self._ranked(query) if product_id in stocked]
        return StagedResults(available=available[:limit], final=available[:limit])

    def _ranked(self, query: str) -> list[str]:
        terms = set(tokenize(query))
        scored = [
            (len(terms & document), product_id)
            for product_id, document in self._documents
            if terms & document
        ]
        # Higher overlap first; product id breaks ties so results are deterministic.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [product_id for _, product_id in scored]


def _overlap_document(product: Product) -> str:
    return " ".join([product.name, product.brand or "", *product.category_path])


class BM25Pipeline:
    """BM25F over product fields. Store stock only restricts eligibility (``allowed``)."""

    def __init__(
        self,
        products: Iterable[Product],
        in_stock: Mapping[str, frozenset[str]],
        settings: BM25Settings,
    ) -> None:
        self._index = BM25FIndex.build(products, settings)
        self._in_stock = in_stock
        self.name = f"bm25f-{bm25_config_id(settings)}"

    def retrieve(self, query: str, depth: int) -> Sequence[str]:
        return [candidate.product_id for candidate in self._index.search(query, depth)]

    def search(self, query: str, store_id: str, limit: int) -> StagedResults:
        allowed = self._in_stock.get(store_id, frozenset())
        available = [c.product_id for c in self._index.search(query, limit, allowed=allowed)]
        return StagedResults(available=available, final=available)


AVAILABLE_ZERO_EXACT = (
    "in-stock EXACT products (v2): a store counts as available-zero when none of its in-stock "
    "results satisfies every constraint, even if RELAXED results are still shown (final-zero "
    "counts those)"
)


@dataclass(frozen=True, slots=True)
class OrderedResults:
    """Constraint-first results for one query and (optional) store, with candidate counts."""

    scored: tuple[ScoredProduct, ...]  # EXACT first, then RELAXED, each in BM25 order
    # Eligible products with a score above zero before the depth cut; None unless requested
    # (counting them costs a full search, so the search path never does).
    eligible_matches: int | None = None

    @property
    def exact(self) -> list[str]:
        return [s.product_id for s in self.scored if s.tier is MatchTier.EXACT]

    @property
    def relaxed(self) -> list[str]:
        return [s.product_id for s in self.scored if s.tier is MatchTier.RELAXED]


class UnderstandingPipeline:
    """BM25F over the concepts produced by query understanding (Phase 4 stages s1-s6).

    With ``stages.ordering`` (s6) each store's search takes its ``candidate_depth`` best
    eligible BM25F candidates and puts EXACT ones (every stated constraint satisfied) before
    RELAXED ones; ``available`` is then the EXACT list. ``retrieve`` is always plain BM25F.
    """

    def __init__(
        self,
        products: Iterable[Product],
        in_stock: Mapping[str, frozenset[str]],
        bm25: BM25Settings,
        query: QuerySettings,
        stage: str,
        stages: Stages,
        ranking: RankingSettings | None = None,
    ) -> None:
        products = list(products)
        self._products = {product.product_id: product for product in products}
        self._index = BM25FIndex.build(products, bm25)
        self._parser = build_parser(products, self._index.vocabulary, query, stages)
        self._in_stock = in_stock
        self._parsed: dict[str, ParsedQuery] = {}
        self._ordering = stages.ordering
        ranking = ranking or RankingSettings()
        self._candidate_depth = ranking.candidate_depth
        self._coverage = (
            CoverageIndex(products, ranking.coverage_fields) if stages.ordering else None
        )
        payload = {"bm25": bm25_config_id(bm25), **parser_identity(query, stages)}
        if stages.ordering:
            payload["candidate_depth"] = self._candidate_depth
            payload["coverage"] = {
                "rule": "all-slots-of-an-alternative-v2",
                "fields": list(ranking.coverage_fields),
            }
        identity = json.dumps(payload, sort_keys=True)
        self.name = f"qu-{stage}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        self.available_zero_definition = AVAILABLE_ZERO_EXACT if stages.ordering else None

    def parse(self, query: str) -> ParsedQuery:
        if query not in self._parsed:
            self._parsed[query] = self._parser.parse(query)
        return self._parsed[query]

    def retrieve(self, query: str, depth: int) -> Sequence[str]:
        concepts = self.parse(query).concepts
        return [c.product_id for c in self._index.search_concepts(concepts, depth)]

    def search(self, query: str, store_id: str, limit: int) -> StagedResults:
        allowed = self._in_stock.get(store_id, frozenset())
        if self._ordering:
            ordered = self.ordered(query, allowed)
            final = [s.product_id for s in ordered.scored][:limit]
            return StagedResults(available=ordered.exact[:limit], final=final)
        concepts = self.parse(query).concepts
        found = self._index.search_concepts(concepts, limit, allowed=allowed)
        available = [candidate.product_id for candidate in found]
        return StagedResults(available=available, final=available)

    def ordered(
        self, query: str, allowed: Collection[str] | None = None, *, count_matches: bool = False
    ) -> OrderedResults:
        """Constraint-first ordering of the ``candidate_depth`` best eligible candidates;
        ``allowed=None`` means the whole catalog."""
        parsed = self.parse(query)
        candidates = self._index.search_concepts(
            parsed.concepts, self._candidate_depth, allowed=allowed
        )
        scored = self._order(parsed, candidates)
        matches = None
        if count_matches:
            matches = (
                len(candidates)
                if len(candidates) < self._candidate_depth
                else self._index.match_count(parsed.concepts, allowed)
            )
        return OrderedResults(tuple(scored), matches)

    def beyond_depth_exact(self, query: str, allowed: Collection[str] | None = None) -> int:
        """EXACT products ranked past ``candidate_depth`` (which S6 cannot promote). For
        evaluation only: it scores every match."""
        parsed = self.parse(query)
        everything = self._index.search_concepts(
            parsed.concepts, max(len(self._products), 1), allowed=allowed
        )
        beyond = everything[self._candidate_depth :]
        if not beyond:
            return 0
        return sum(s.tier is MatchTier.EXACT for s in self._order(parsed, beyond))

    def _order(self, parsed: ParsedQuery, candidates: Sequence[Candidate]) -> list[ScoredProduct]:
        if self._coverage is None:
            raise RuntimeError("constraint ordering needs the ordering stage (s6)")
        ids = [candidate.product_id for candidate in candidates]
        coverage = self._coverage.covered(parsed.concepts, ids)
        required = self._coverage.coverable(parsed.concepts) - set(parsed.brand_concepts)
        return order(candidates, self._products, parsed, coverage, required)
