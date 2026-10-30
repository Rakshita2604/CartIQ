"""In-memory BM25F lexical retrieval over product fields.

Variant implemented: BM25F with per-field length normalization and a single saturation
step (Robertson, Zaragoza & Taylor, "Simple BM25 extension to multiple weighted fields",
CIKM 2004), using Lucene's non-negative IDF::

    B_f(d)      = 1 - b_f + b_f * len_f(d) / avglen_f
    tf~(t, d)   = sum over indexed fields f of  w_f * tf_f(t, d) / B_f(d)
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = sum over t in U(q) of  idf(t) * tf~(t, d) / (k1 + tf~(t, d))

Definitions:

- Indexed fields are those with weight w_f > 0. Fields with weight 0 are dropped before
  any statistic is computed: they affect no score, no df and no average length.
- N is the number of documents. df(t) counts documents where t occurs in at least one
  indexed field. len_f(d) is the token count of field f in d; avglen_f is the mean of
  len_f over documents whose field f is non-empty.
- U(q) is the set of unique query tokens. Tokens absent from the index are ignored.
- The classic (k1 + 1) numerator factor is intentionally omitted, as in Lucene 8+. It
  multiplies every term contribution by the same positive constant, so omitting it changes
  the absolute score scale but never the ranking: each term contributes less than its idf.
  With a single field, w = 1, the score equals classic BM25 divided by (k1 + 1). Compare
  BM25F scores only with each other, not with scores from other BM25 implementations.

Concepts (``search_concepts``). A query can group alternative terms into one concept (a
typed word, its lexicon expansions, a spelling correction, plural variants). Each term
carries a weight w_t in (0, 1]. A concept is scored like a single term, saturating once:

    tf~_C(d)    = sum over t in C of  w_t * tf~(t, d)
    df_C        = number of documents with tf~_C(d) > 0  (the union of the terms' documents)
    score(q, d) = sum over concepts C of  idf(df_C) * tf~_C(d) / (k1 + tf~_C(d))

so matching several alternatives of one concept can never contribute more than idf(df_C).
A concept with a single term of weight 1 is exactly the term formula above; ``search``
builds one such concept per unique query token, in sorted order, and is unchanged from the
original single-term implementation.

Weighted term frequencies tf~ are computed once at build time; idf and saturation are
applied at query time, concept by concept in the given order, so scores are bit-identical
across runs. Results are the documents with score > 0, sorted by score descending, then
product id ascending. ``explain`` reports each concept's contribution for one product.

``allowed`` restricts which products may be returned. It is applied after scoring and
before the result limit, and never influences idf, lengths or scores. Future optimization
(deliberately not done yet): the eligibility mask is rebuilt from ``allowed`` on every call,
which dominates store-filtered latency on large catalogs (about 1 ms of a 1 ms query on the
27k-product BigBasket snapshot); caching one mask per store and stock version would remove it.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

import numpy as np
import numpy.typing as npt

from cartiq.config import BM25Settings, FieldWeight
from cartiq.domain import Candidate, Product, QueryConcept
from cartiq.parsing.text import TOKENIZER_VERSION, tokenize
from cartiq.retrieval.documents import Field, product_fields

FORMULA_VERSION = "bm25f-v1"


def bm25_config_id(settings: BM25Settings) -> str:
    """Short identity of everything besides the catalog that determines BM25F results."""
    payload = json.dumps(
        {
            "formula": FORMULA_VERSION,
            "tokenizer": TOKENIZER_VERSION,
            "settings": settings.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class _Postings:
    documents: npt.NDArray[np.intp]  # ascending document indices
    weighted_tf: npt.NDArray[np.float64]  # tf~(t, d) for those documents, all > 0


@dataclass(frozen=True, slots=True)
class TermExplanation:
    term: str
    weight: float
    weighted_tf: float  # tf~(t, d); 0 when the product does not contain the term


@dataclass(frozen=True, slots=True)
class ConceptExplanation:
    text: str
    document_frequency: int  # df_C, the union over the concept's terms
    idf: float
    weighted_tf: float  # tf~_C(d)
    contribution: float  # idf * tf~_C / (k1 + tf~_C); 0 when the concept does not match
    terms: tuple[TermExplanation, ...]


class BM25FIndex:
    def __init__(
        self,
        product_ids: Sequence[str],
        postings: Mapping[str, _Postings],
        average_lengths: Mapping[Field, float],
        k1: float,
    ) -> None:
        self._product_ids = tuple(product_ids)
        self._positions = {product_id: i for i, product_id in enumerate(self._product_ids)}
        self._postings = postings
        self._average_lengths = average_lengths
        self._k1 = k1

    @classmethod
    def build(cls, products: Iterable[Product], settings: BM25Settings) -> "BM25FIndex":
        products = list(products)
        return cls.from_documents(
            [product.product_id for product in products],
            [product_fields(product) for product in products],
            settings,
        )

    @classmethod
    def from_documents(
        cls,
        product_ids: Sequence[str],
        documents: Sequence[Mapping[Field, Sequence[str]]],
        settings: BM25Settings,
    ) -> "BM25FIndex":
        if len(product_ids) != len(documents):
            raise ValueError("product_ids and documents differ in length")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("duplicate product ids")
        # Document order (and so tie-breaking) is product id order, whatever the input order.
        ordered = sorted(zip(product_ids, documents, strict=True), key=lambda pair: pair[0])
        ids = [product_id for product_id, _ in ordered]
        docs = [document for _, document in ordered]

        fields = {
            field: weight
            for field in Field
            if (weight := _field_weight(settings, field)).weight > 0
        }
        average_lengths = {field: _average_length(docs, field) for field in fields}

        weighted_tf: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        for position, document in enumerate(docs):
            for term, value in _weighted_tf(document, fields, average_lengths).items():
                weighted_tf[term].append((position, value))

        postings = {
            term: _Postings(
                documents=np.array([position for position, _ in entries], dtype=np.intp),
                weighted_tf=np.array([value for _, value in entries], dtype=np.float64),
            )
            for term, entries in sorted(weighted_tf.items())
        }
        return cls(ids, postings, average_lengths, settings.k1)

    def __len__(self) -> int:
        return len(self._product_ids)

    @property
    def term_count(self) -> int:
        return len(self._postings)

    def idf(self, term: str) -> float | None:
        postings = self._postings.get(term)
        return None if postings is None else self._idf(len(postings.documents))

    def document_frequency(self, term: str) -> int:
        postings = self._postings.get(term)
        return 0 if postings is None else len(postings.documents)

    @property
    def vocabulary(self) -> frozenset[str]:
        return frozenset(self._postings)

    def average_length(self, field: Field) -> float | None:
        """Mean token count of an indexed field over documents that have it; None if the
        field is not indexed."""
        return self._average_lengths.get(field)

    def search(
        self, query: str, depth: int, allowed: Collection[str] | None = None
    ) -> list[Candidate]:
        """Plain BM25F: one single-term concept per unique query token, in sorted order."""
        concepts = [QueryConcept.single(term) for term in sorted(set(tokenize(query)))]
        return self.search_concepts(concepts, depth, allowed)

    def search_concepts(
        self, concepts: Sequence[QueryConcept], depth: int, allowed: Collection[str] | None = None
    ) -> list[Candidate]:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        scores = np.zeros(len(self._product_ids), dtype=np.float64)
        matched = False
        for concept in concepts:
            folded = self._fold(concept)
            if folded is None:
                continue
            documents, tf = folded
            scores[documents] += self._idf(len(documents)) * tf / (self._k1 + tf)
            matched = True
        if not matched:
            return []

        eligible = np.flatnonzero(scores > 0)
        if allowed is not None:
            mask = np.zeros(len(self._product_ids), dtype=bool)
            mask[[self._positions[pid] for pid in allowed if pid in self._positions]] = True
            eligible = eligible[mask[eligible]]

        # lexsort uses the last key first: score descending, then position (= product id).
        order = np.lexsort((eligible, -scores[eligible]))
        top = eligible[order[:depth]]
        return [
            Candidate(
                product_id=self._product_ids[position],
                fused_score=float(scores[position]),
                bm25_score=float(scores[position]),
                bm25_rank=rank,
            )
            for rank, position in enumerate(top.tolist(), start=1)
        ]

    def explain(
        self, concepts: Sequence[QueryConcept], product_id: str
    ) -> tuple[ConceptExplanation, ...]:
        """Per-concept score contributions for one product (for debugging, not ranking)."""
        position = self._positions[product_id]
        explanations = []
        for concept in concepts:
            folded = self._fold(concept)
            df = 0 if folded is None else len(folded[0])
            terms = tuple(
                TermExplanation(t.term, t.weight, self._term_tf(t.term, position))
                for t in concept.terms
            )
            tf = 0.0
            if folded is not None:
                documents, values = folded
                hit = np.searchsorted(documents, position)
                if hit < len(documents) and documents[hit] == position:
                    tf = float(values[hit])
            idf = self._idf(df) if df else 0.0
            explanations.append(
                ConceptExplanation(
                    text=concept.text,
                    document_frequency=df,
                    idf=idf,
                    weighted_tf=tf,
                    contribution=idf * tf / (self._k1 + tf) if tf else 0.0,
                    terms=terms,
                )
            )
        return tuple(explanations)

    def _fold(
        self, concept: QueryConcept
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]] | None:
        """Documents matching a concept and their tf~_C, or None when nothing matches."""
        present = [
            (t.weight, self._postings[t.term]) for t in concept.terms if t.term in self._postings
        ]
        if not present:
            return None
        if len(present) == 1 and present[0][0] == 1.0:
            postings = present[0][1]
            return postings.documents, postings.weighted_tf
        accumulated = np.zeros(len(self._product_ids), dtype=np.float64)
        for weight, postings in present:
            accumulated[postings.documents] += weight * postings.weighted_tf
        documents = np.flatnonzero(accumulated > 0)
        return documents, accumulated[documents]

    def _term_tf(self, term: str, position: int) -> float:
        postings = self._postings.get(term)
        if postings is None:
            return 0.0
        hit = np.searchsorted(postings.documents, position)
        if hit < len(postings.documents) and postings.documents[hit] == position:
            return float(postings.weighted_tf[hit])
        return 0.0

    def _idf(self, document_frequency: int) -> float:
        n = len(self._product_ids)
        return math.log(1 + (n - document_frequency + 0.5) / (document_frequency + 0.5))


def _field_weight(settings: BM25Settings, field: Field) -> FieldWeight:
    weight: FieldWeight = getattr(settings, field.value)
    return weight


def _average_length(documents: Sequence[Mapping[Field, Sequence[str]]], field: Field) -> float:
    lengths = [len(tokens) for document in documents if (tokens := document.get(field))]
    return fmean(lengths) if lengths else 0.0


def _weighted_tf(
    document: Mapping[Field, Sequence[str]],
    fields: Mapping[Field, FieldWeight],
    average_lengths: Mapping[Field, float],
) -> dict[str, float]:
    """tf~(t, d) for every term of one document, summing fields in a fixed order."""
    weighted: dict[str, float] = {}
    for field, spec in fields.items():
        tokens = document.get(field)
        if not tokens:
            continue
        normalizer = 1 - spec.b + spec.b * len(tokens) / average_lengths[field]
        for term, count in sorted(Counter(tokens).items()):
            weighted[term] = weighted.get(term, 0.0) + spec.weight * count / normalizer
    return weighted
