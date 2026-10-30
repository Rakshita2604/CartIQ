"""Turn a raw query into a ParsedQuery of retrieval concepts.

Stages, each switchable so it can be evaluated on its own (S1-S4 cumulatively):

1. quantity    extract the requested quantity; remove its text from retrieval
2. lexicon     expand Hinglish words/phrases with English equivalents (longest phrase first)
3. spelling    correct unknown tokens; add dominant neighbours of rare known tokens
4. inflection  add singular/plural forms that the catalog uses

Every concept keeps the typed term. Expansions never replace it, and a lexicon phrase or
correction is one concept, so synonyms saturate together in BM25F. Within a lexicon
concept each alternative (the typed phrase, each English phrase) carries total weight 1,
split evenly over its words: "cold drink" gives "cold" and "drink" 0.5 each, so a product
matching just one weakly related word ("Mango *Drink*") gets less credit than one that
matches the concept's own word.
"""

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from cartiq.config import QuerySettings
from cartiq.domain import ConceptTerm, Correction, ParsedQuery, QueryConcept, TermSource
from cartiq.parsing.text import tokenize
from cartiq.query.inflection import variants
from cartiq.query.lexicon import CONNECTORS, Lexicon, LexiconEntry
from cartiq.query.quantities import extract_quantity
from cartiq.query.spelling import SpellingCorrector

PARSER_VERSION = "qu-v1"


@dataclass(frozen=True, slots=True)
class Stages:
    quantity: bool = False
    lexicon: bool = False
    spelling: bool = False
    inflection: bool = False


STAGES = {
    "s1": Stages(quantity=True),
    "s2": Stages(quantity=True, lexicon=True),
    "s3": Stages(quantity=True, lexicon=True, spelling=True),
    "s4": Stages(quantity=True, lexicon=True, spelling=True, inflection=True),
}


class QueryParser:
    def __init__(
        self,
        document_frequency: Mapping[str, int],
        lexicon: Lexicon,
        settings: QuerySettings,
        stages: Stages,
    ) -> None:
        """``document_frequency`` is the retrieval index's vocabulary with document counts."""
        self._vocabulary = frozenset(document_frequency)
        self._lexicon = lexicon
        self._settings = settings
        self._stages = stages
        self._speller = (
            SpellingCorrector(document_frequency, lexicon.single_terms, settings)
            if stages.spelling
            else None
        )

    def parse(self, raw: str) -> ParsedQuery:
        text = unicodedata.normalize("NFKC", raw)
        mention = None
        retrieval_text = " ".join(text.split())
        if self._stages.quantity:
            extraction = extract_quantity(text, bare_count_max=self._settings.bare_count_max)
            mention, retrieval_text = extraction.mention, extraction.retrieval_text
        tokens = tokenize(retrieval_text)

        concepts: list[QueryConcept] = []
        corrections: list[Correction] = []
        position = 0
        while position < len(tokens):
            match = self._lexicon.longest_match(tokens, position) if self._stages.lexicon else None
            if match is not None:
                length, entry = match
                phrase = tokens[position : position + length]
                concepts.append(_lexicon_concept(" ".join(phrase), [tuple(phrase)], entry))
                position += length
                continue
            token = tokens[position]
            concepts.append(self._token_concept(token, corrections))
            position += 1

        if self._stages.inflection:
            concepts = [self._with_variants(concept) for concept in concepts]
        # Like plain BM25F: a repeated word counts once, and concepts are scored in sorted
        # order, so with every stage off the parser reproduces ``BM25FIndex.search``.
        unique: dict[str, QueryConcept] = {}
        for concept in concepts:
            unique.setdefault(concept.text, concept)
        concepts = [unique[text] for text in sorted(unique)]
        return ParsedQuery(
            raw=raw,
            normalized=retrieval_text,
            tokens=tuple(tokens),
            concepts=tuple(concepts),
            corrections=tuple(corrections),
            quantity_mention=mention,
            quantity=mention.quantity if mention else None,
            pack_count=mention.pack_count if mention else None,
        )

    def _token_concept(self, token: str, corrections: list[Correction]) -> QueryConcept:
        correction = self._speller.correct(token) if self._speller else None
        if correction is None:
            return QueryConcept.single(token)
        corrections.append(correction)
        entry = self._lexicon.entry((correction.corrected,)) if self._stages.lexicon else None
        if entry is not None:
            # Corrected to a lexicon term ("dhanya" -> "dhaniya"): expand it as well.
            return _lexicon_concept(
                token, [(token,), (correction.corrected,)], entry, TermSource.SPELLING
            )
        return QueryConcept(
            token,
            (
                ConceptTerm(token, TermSource.QUERY),
                ConceptTerm(correction.corrected, TermSource.SPELLING),
            ),
        )

    def _with_variants(self, concept: QueryConcept) -> QueryConcept:
        terms = list(concept.terms)
        present = {term.term for term in terms}
        for term in concept.terms:
            for form in variants(term.term, self._vocabulary):
                if form not in present:
                    terms.append(ConceptTerm(form, TermSource.INFLECTION, term.weight))
                    present.add(form)
        return QueryConcept(concept.text, tuple(terms))


def _lexicon_concept(
    text: str,
    typed: list[tuple[str, ...]],
    entry: LexiconEntry,
    second_source: TermSource = TermSource.QUERY,
) -> QueryConcept:
    """Concept for a lexicon match. ``typed`` holds the shopper's phrase first, then
    (for a spelling correction) the corrected lexicon term."""
    weights: dict[str, tuple[float, TermSource]] = {}

    def add(phrase: tuple[str, ...], source: TermSource) -> None:
        words = [word for word in phrase if word not in CONNECTORS] or list(phrase)
        for word in words:
            weight = 1 / len(words)
            current = weights.get(word)
            if current is None or weight > current[0]:
                # Keep the earliest source for a word already present at the same weight.
                weights[word] = (weight, current[1] if current else source)

    add(typed[0], TermSource.QUERY)
    for extra in typed[1:]:
        add(extra, second_source)
    for phrase in entry.english:
        add(phrase, TermSource.LEXICON)
    return QueryConcept(
        text, tuple(ConceptTerm(word, source, weight) for word, (weight, source) in weights.items())
    )
