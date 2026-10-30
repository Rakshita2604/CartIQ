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

Each (term, document) contribution is computed once at build time. A query adds them up
term by term in sorted order, so scores are bit-identical across runs. Results are the
documents with score > 0, sorted by score descending, then product id ascending.

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
from cartiq.domain import Candidate, Product
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
    contributions: npt.NDArray[np.float64]  # idf(t) * tf~ / (k1 + tf~), all > 0


class BM25FIndex:
    def __init__(
        self,
        product_ids: Sequence[str],
        postings: Mapping[str, _Postings],
        idf: Mapping[str, float],
        average_lengths: Mapping[Field, float],
    ) -> None:
        self._product_ids = tuple(product_ids)
        self._positions = {product_id: i for i, product_id in enumerate(self._product_ids)}
        self._postings = postings
        self._idf = idf
        self._average_lengths = average_lengths

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

        n = len(docs)
        postings: dict[str, _Postings] = {}
        idf: dict[str, float] = {}
        for term in sorted(weighted_tf):
            entries = weighted_tf[term]
            term_idf = math.log(1 + (n - len(entries) + 0.5) / (len(entries) + 0.5))
            idf[term] = term_idf
            tf = np.array([value for _, value in entries], dtype=np.float64)
            postings[term] = _Postings(
                documents=np.array([position for position, _ in entries], dtype=np.intp),
                contributions=term_idf * tf / (settings.k1 + tf),
            )
        return cls(ids, postings, idf, average_lengths)

    def __len__(self) -> int:
        return len(self._product_ids)

    @property
    def term_count(self) -> int:
        return len(self._postings)

    def idf(self, term: str) -> float | None:
        return self._idf.get(term)

    def average_length(self, field: Field) -> float | None:
        """Mean token count of an indexed field over documents that have it; None if the
        field is not indexed."""
        return self._average_lengths.get(field)

    def search(
        self, query: str, depth: int, allowed: Collection[str] | None = None
    ) -> list[Candidate]:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        postings = [self._postings[t] for t in sorted(set(tokenize(query))) if t in self._postings]
        if not postings:
            return []

        scores = np.zeros(len(self._product_ids), dtype=np.float64)
        for posting in postings:
            scores[posting.documents] += posting.contributions
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
