"""Which query concepts a product contains, for exact matching (stage s6).

Coverage is decided on a fixed set of product fields (by default name, brand and product
type), separately from retrieval: shelf text (sub-category, top-level category) and
descriptions can still make BM25F retrieve and score a product, but cannot make it an exact
match for a concept. A product contains a concept when every word slot of at least one of
the concept's alternatives (``QueryConcept.phrases``) has a form among the product's words
in those fields.
"""

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence

from cartiq.domain import Alternative, Product, QueryConcept
from cartiq.retrieval.documents import Field, product_fields

DEFAULT_COVERAGE_FIELDS = ("name", "brand", "product_type")


class CoverageIndex:
    def __init__(self, products: Iterable[Product], fields: Collection[str]) -> None:
        selected = [Field(name) for name in fields]
        self._words: dict[str, frozenset[str]] = {}
        postings: defaultdict[str, set[str]] = defaultdict(set)
        for product in products:
            document = product_fields(product)
            words = frozenset(token for field in selected for token in document[field])
            self._words[product.product_id] = words
            for word in words:
                postings[word].add(product.product_id)
        self._postings = {word: frozenset(ids) for word, ids in postings.items()}

    def covered(
        self, concepts: Sequence[QueryConcept], product_ids: Sequence[str]
    ) -> dict[str, frozenset[str]]:
        """For each product, the texts of the concepts it contains."""
        return {
            pid: frozenset(
                concept.text
                for concept in concepts
                if any(_contains(self._words[pid], alt) for alt in concept.phrases)
            )
            for pid in product_ids
        }

    def coverable(self, concepts: Sequence[QueryConcept]) -> frozenset[str]:
        """Texts of the concepts at least one catalog product contains. A concept no product
        can contain is not required, so it cannot make every candidate non-exact."""
        return frozenset(
            concept.text
            for concept in concepts
            if any(self._products_containing(alt) for alt in concept.phrases)
        )

    def _products_containing(self, alternative: Alternative) -> frozenset[str]:
        found: frozenset[str] | None = None
        for slot in alternative:
            ids = frozenset().union(*(self._postings.get(form, frozenset()) for form in slot))
            found = ids if found is None else found & ids
            if not found:
                return frozenset()
        return found or frozenset()


def _contains(words: frozenset[str], alternative: Alternative) -> bool:
    return all(slot & words for slot in alternative)
