"""Singular/plural variants of a term, restricted to forms the catalog actually uses.

Light, query-side only: "masala" also matches "masalas" when the catalog has "masalas".
The index is untouched (no stemming), so results differ only for queries that use it.
"""

from collections.abc import Collection

_MIN_LENGTH = 3


def variants(term: str, vocabulary: Collection[str]) -> list[str]:
    """Other forms of ``term`` (adding or removing a final "s"/"es") that are in the
    vocabulary, sorted."""
    if len(term) < _MIN_LENGTH or not term.isalpha():
        return []
    forms = {term + "s", term + "es"}
    if term.endswith("es") and len(term) > _MIN_LENGTH + 1:
        forms.add(term[:-2])
    if term.endswith("s") and len(term) > _MIN_LENGTH:
        forms.add(term[:-1])
    return sorted(form for form in forms if form != term and form in vocabulary)
