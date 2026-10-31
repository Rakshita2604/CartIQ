"""Spelling correction against the catalog vocabulary (and lexicon terms).

Symmetric-delete lookup (as in SymSpell): every target term is indexed under the strings
obtained by deleting up to two of its characters; a query token's deletes find candidate
terms, which are then verified with the optimal-string-alignment distance (insert, delete,
substitute, or swap two adjacent characters, each costing 1).

Rules, fixed in the Phase 4 design:

- Tokens shorter than ``one_edit_min_length``, tokens containing a digit, and lexicon terms
  are never corrected.
- Allowed distance: 1 from ``one_edit_min_length``, 2 from ``two_edit_min_length``.
- Unknown token (not a target term): the nearest target wins. Among targets at that same
  minimum distance, ``preferred`` terms (catalog-brand words, Phase 4b) come first; then,
  when ``phonetic`` is on, the target that sounds most like the token (Soundex code
  distance, see ``phonetic``); then higher document frequency; then alphabetical order.
- Known but rare token (real catalogs contain misspellings, e.g. "panner"): a one-edit
  neighbour is added only if it dominates: df >= ``dominance_ratio`` x the token's df and
  df >= ``dominance_min_df``.

Corrections are always added alongside the typed token, never replacing it, and are
recorded on the parsed query.

The target vocabulary comes from ``spelling_frequencies``: by default only product name,
brand and product type. Words that occur only in descriptions or category text ("comes
with ...") are never targets, and their frequencies cannot make a neighbour "dominant".
"""

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping

from cartiq.config import QuerySettings
from cartiq.domain import Correction, CorrectionReason, Product
from cartiq.query.phonetic import phonetic_distance
from cartiq.query.spelling_distance import osa_distance
from cartiq.retrieval.documents import Field, product_fields


def spelling_frequencies(products: Iterable[Product], fields: Collection[str]) -> dict[str, int]:
    """Document frequency of each word over the given product fields only."""
    selected = [Field(name) for name in fields]
    counts: Counter[str] = Counter()
    for product in products:
        document = product_fields(product)
        counts.update({token for field in selected for token in document[field]})
    return dict(counts)


class SpellingCorrector:
    def __init__(
        self,
        document_frequency: Mapping[str, int],
        protected: Iterable[str],
        settings: QuerySettings,
        preferred: Iterable[str] = (),
        *,
        phonetic: bool = False,
    ) -> None:
        """``document_frequency`` is the target vocabulary (see ``spelling_frequencies``);
        ``protected`` terms (the lexicon's) are valid targets with df 0 and are never
        corrected themselves."""
        self._df = dict(document_frequency)
        self._protected = frozenset(protected)
        self._preferred = frozenset(preferred)
        self._phonetic = phonetic
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
                (
                    distance,
                    term not in self._preferred,
                    phonetic_distance(token, term) if self._phonetic else 0,
                    -self._df.get(term, 0),
                    term,
                )
                for term in self._candidates(token, max_distance)
                if (distance := osa_distance(token, term)) <= max_distance
            ),
            default=None,
        )
        if best is None:
            return None
        return Correction(token, best[-1], best[0], CorrectionReason.UNKNOWN)

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

    def candidates(self, token: str) -> frozenset[str]:
        """Target terms sharing a delete-variant with ``token`` at its allowed distance,
        before distance verification (the lookup's work, for measurement and tests)."""
        return frozenset(self._candidates(token, self._max_distance(len(token))))

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


def _deletes(term: str, depth: int) -> set[str]:
    """``term`` and every string reachable by deleting up to ``depth`` characters."""
    level = {term}
    found = {term}
    for _ in range(depth):
        level = {word[:i] + word[i + 1 :] for word in level for i in range(len(word))}
        found |= level
    return found
