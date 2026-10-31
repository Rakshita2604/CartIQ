"""Exact dense retrieval over an embedded catalog (Phase 5A; separate from BM25F).

Vectors are L2-normalised, so a matrix product gives cosine similarities. Store
eligibility (``allowed``) is applied before the top-k cut. Results keep the raw cosine;
for ordering only, scores are rounded to ``SORT_DECIMALS`` decimals (so floating-point
noise from batching or BLAS threads cannot reorder results) and ties go to the lower
product id.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

SORT_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class DenseHit:
    product_id: str
    score: float  # raw cosine similarity
    rank: int


class DenseIndex:
    def __init__(self, product_ids: Sequence[str], vectors: npt.NDArray[np.float32]) -> None:
        if len(product_ids) != vectors.shape[0]:
            raise ValueError("one vector per product id is required")
        order = sorted(range(len(product_ids)), key=lambda i: product_ids[i])
        self._ids = tuple(product_ids[i] for i in order)
        self._positions = {pid: i for i, pid in enumerate(self._ids)}
        self._vectors = np.ascontiguousarray(vectors[order], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self,
        query: npt.NDArray[np.float32],
        depth: int,
        allowed: Collection[str] | None = None,
    ) -> list[DenseHit]:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        scores = self._vectors @ query.astype(np.float32)
        if allowed is None:
            eligible = np.arange(len(self._ids))
        else:
            eligible = np.array(
                sorted(self._positions[pid] for pid in allowed if pid in self._positions),
                dtype=np.intp,
            )
        if eligible.size == 0:
            return []
        rounded = np.round(scores[eligible].astype(np.float64), SORT_DECIMALS)
        order = np.lexsort((eligible, -rounded))  # score descending, then product id
        top = eligible[order[:depth]]
        return [
            DenseHit(self._ids[i], float(scores[i]), rank)
            for rank, i in enumerate(top.tolist(), start=1)
        ]
