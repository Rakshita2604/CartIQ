"""Spelling correction against the catalog vocabulary (and lexicon terms).

Symmetric-delete lookup (as in SymSpell): every target term is indexed under the strings
obtained by deleting up to two of its characters; a query token's deletes find candidate
terms, which are then verified with the optimal-string-alignment distance (insert, delete,
substitute, or swap two adjacent characters, each costing 1).

Rules, fixed in the Phase 4 design:

- Tokens shorter than ``one_edit_min_length``, tokens containing a digit, and lexicon terms
  are never corrected.
- Allowed distance: 1 from ``one_edit_min_length``, 2 from ``two_edit_min_length``.
- Unknown token (not a target term): the nearest target wins, ties broken by higher
  document frequency, then alphabetically.
- Known but rare token (real catalogs contain misspellings, e.g. "panner"): a one-edit
  neighbour is added only if it dominates: df >= ``dominance_ratio`` x the token's df and
  df >= ``dominance_min_df``.

Corrections are always added alongside the typed token, never replacing it, and are
recorded on the parsed query.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from cartiq.config import QuerySettings
from cartiq.domain import Correction, CorrectionReason


class SpellingCorrector:
    def __init__(
        self,
        document_frequency: Mapping[str, int],
        protected: Iterable[str],
        settings: QuerySettings,
    ) -> None:
        """``document_frequency`` covers the catalog vocabulary; ``protected`` terms (the
        lexicon's) are valid targets with df 0 and are never corrected themselves."""
        self._df = dict(document_frequency)
        self._protected = frozenset(protected)
        self._settings = settings
        self._deletes: defaultdict[str, set[str]] = defaultdict(set)
        for term in sorted(self._df.keys() | self._protected):
            for variant in _deletes(term, self._index_depth(len(term))):
                self._deletes[variant].add(term)

    def correct(self, token: str) -> Correction | None:
        if token in self._protected or any(ch.isdigit() for ch in token):
            return None
        max_distance = self._max_distance(len(token))
        if max_distance == 0:
            return None
        if token in self._df:
            return self._dominant_neighbour(token)
        best = min(
            (
                (distance, -self._df.get(term, 0), term)
                for term in self._candidates(token, max_distance)
                if (distance := osa_distance(token, term)) <= max_distance
            ),
            default=None,
        )
        if best is None:
            return None
        return Correction(token, best[2], best[0], CorrectionReason.UNKNOWN)

    def _dominant_neighbour(self, token: str) -> Correction | None:
        floor = max(
            self._settings.dominance_ratio * self._df[token], self._settings.dominance_min_df
        )
        dominant = [
            (-self._df[term], term)
            for term in self._candidates(token, 1)
            if term != token
            and term in self._df
            and self._df[term] >= floor
            and osa_distance(token, term) == 1
        ]
        if not dominant:
            return None
        return Correction(token, min(dominant)[1], 1, CorrectionReason.DOMINANT)

    def _candidates(self, token: str, max_distance: int) -> set[str]:
        found: set[str] = set()
        for variant in _deletes(token, max_distance):
            found |= self._deletes.get(variant, set())
        return found

    def _max_distance(self, length: int) -> int:
        if length >= self._settings.two_edit_min_length:
            return 2
        if length >= self._settings.one_edit_min_length:
            return 1
        return 0

    def _index_depth(self, length: int) -> int:
        """Deletes to index for a target of this length: as many as any query token that
        could reach it within its allowed distance would use."""
        if length >= self._settings.two_edit_min_length - 2:
            return 2
        if length >= self._settings.one_edit_min_length - 1:
            return 1
        return 0


def osa_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (Damerau-Levenshtein without repeated edits)."""
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i, *([0] * len(b))]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                current[j] = min(current[j], previous2[j - 2] + 1)
        previous2, previous = previous, current
    return previous[-1]


def _deletes(term: str, depth: int) -> set[str]:
    """``term`` and every string reachable by deleting up to ``depth`` characters."""
    level = {term}
    found = {term}
    for _ in range(depth):
        level = {word[:i] + word[i + 1 :] for word in level for i in range(len(word))}
        found |= level
    return found
