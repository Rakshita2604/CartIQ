"""Search pipelines that can be evaluated. Each exposes whole-catalog retrieval (ignoring
stock) and per-store staged results. Neither has fallbacks or substitutions yet, so final
results are the in-stock matches."""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict

from cartiq.config import BM25Settings, QuerySettings
from cartiq.domain import ParsedQuery, Product
from cartiq.evaluation.runner import StagedResults
from cartiq.parsing.text import tokenize
from cartiq.query import QueryParser, Stages
from cartiq.query.lexicon import load_lexicon
from cartiq.query.parser import PARSER_VERSION
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


class UnderstandingPipeline:
    """BM25F over the concepts produced by query understanding (Phase 4a stages)."""

    def __init__(
        self,
        products: Iterable[Product],
        in_stock: Mapping[str, frozenset[str]],
        bm25: BM25Settings,
        query: QuerySettings,
        stage: str,
        stages: Stages,
    ) -> None:
        self._index = BM25FIndex.build(products, bm25)
        lexicon = load_lexicon(query.lexicon)
        frequencies = {
            term: self._index.document_frequency(term) for term in self._index.vocabulary
        }
        self._parser = QueryParser(frequencies, lexicon, query, stages)
        self._in_stock = in_stock
        self._parsed: dict[str, ParsedQuery] = {}
        identity = json.dumps(
            {
                "bm25": bm25_config_id(bm25),
                "parser": PARSER_VERSION,
                "lexicon": lexicon.sha256,
                "query": query.model_dump(mode="json"),
                "stages": asdict(stages),
            },
            sort_keys=True,
        )
        self.name = f"qu-{stage}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"

    def parse(self, query: str) -> ParsedQuery:
        if query not in self._parsed:
            self._parsed[query] = self._parser.parse(query)
        return self._parsed[query]

    def retrieve(self, query: str, depth: int) -> Sequence[str]:
        concepts = self.parse(query).concepts
        return [c.product_id for c in self._index.search_concepts(concepts, depth)]

    def search(self, query: str, store_id: str, limit: int) -> StagedResults:
        allowed = self._in_stock.get(store_id, frozenset())
        concepts = self.parse(query).concepts
        found = self._index.search_concepts(concepts, limit, allowed=allowed)
        available = [candidate.product_id for candidate in found]
        return StagedResults(available=available, final=available)
