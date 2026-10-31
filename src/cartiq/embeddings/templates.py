"""Versioned text templates: what text a product or a query becomes before encoding.

Model-specific prefixes are added by the encoder, not here.

Products:
- ``base-v1``: "{brand} {name}. {product type}." The brand is not repeated when the name
  already starts with it. Sub-category, top-level category and description are left out.
- ``desc-v1``: ``base-v1`` followed by the first 32 words of the description.

Queries:
- ``raw-v1``: the query without its quantity text ("amul milk 1 litre" -> "amul milk"),
  using the frozen Phase 4 quantity extraction; the whole query if nothing else remains.
- ``expanded-v1``: ``raw-v1`` followed by the words the frozen Phase 4 parser (stage s6)
  adds through the Hinglish lexicon, spelling corrections and brand aliases (plural forms
  excluded), de-duplicated, in the parser's concept order.
"""

from collections.abc import Callable
from enum import StrEnum

from cartiq.domain import ParsedQuery, Product, TermSource

DESCRIPTION_WORDS = 32


class ProductTemplate(StrEnum):
    BASE = "base-v1"
    DESCRIPTION = "desc-v1"


class QueryTemplate(StrEnum):
    RAW = "raw-v1"
    EXPANDED = "expanded-v1"


def product_text(product: Product, template: ProductTemplate) -> str:
    name = product.name.strip()
    brand = (product.brand or "").strip()
    head = name if not brand or name.casefold().startswith(brand.casefold()) else f"{brand} {name}"
    product_type = product.category_path[-1] if len(product.category_path) > 1 else ""
    text = f"{head}. {product_type}." if product_type else f"{head}."
    if template is ProductTemplate.DESCRIPTION:
        words = product.description.split()[:DESCRIPTION_WORDS]
        if words:
            text = f"{text} {' '.join(words)}"
    return text


_ADDED = frozenset({TermSource.LEXICON, TermSource.SPELLING, TermSource.BRAND})


def query_text(raw: str, parse: Callable[[str], ParsedQuery], template: QueryTemplate) -> str:
    """``parse`` is the frozen stage-s6 parser (only read)."""
    parsed = parse(raw)
    base = parsed.normalized.strip() or raw.strip()
    if template is QueryTemplate.RAW:
        return base
    present = set(base.casefold().split())
    added: list[str] = []
    for concept in parsed.concepts:
        for term in concept.terms:
            if term.source in _ADDED and term.term not in present:
                added.append(term.term)
                present.add(term.term)
    return f"{base} {' '.join(added)}" if added else base
