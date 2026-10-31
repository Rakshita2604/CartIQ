"""Turn a raw query into a ParsedQuery of retrieval concepts.

Stages, each switchable so it can be evaluated on its own (S1-S4 cumulatively):

1. quantity    extract the requested quantity; remove its text from retrieval
2. lexicon     expand Hinglish words/phrases with English equivalents (longest phrase first)
3. spelling    correct unknown tokens; add dominant neighbours of rare known tokens
4. inflection  add singular/plural forms that the catalog uses
5. brands      recognise brands and aliases (before the lexicon and spelling, and again on
               corrected tokens); protect brand/alias words from spelling correction and
               prefer catalog-brand words, then sound-alike words (Soundex), among
               equally close corrections

Every concept keeps the typed term. Expansions never replace it, and a lexicon phrase or
correction is one concept, so synonyms saturate together in BM25F. Within a lexicon
concept each alternative (the typed phrase, each English phrase) carries total weight 1,
split evenly over its words: "cold drink" gives "cold" and "drink" 0.5 each, so a product
matching just one weakly related word ("Mango *Drink*") gets less credit than one that
matches the concept's own word.
"""

import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from cartiq.config import QuerySettings
from cartiq.domain import ConceptTerm, Correction, ParsedQuery, QueryConcept, TermSource
from cartiq.parsing.text import tokenize
from cartiq.query.brands import BrandGazetteer, BrandMatch
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
    brands: bool = False
    ordering: bool = False  # s6: constraint-first ordering (applied after retrieval)


STAGES = {
    "s1": Stages(quantity=True),
    "s2": Stages(quantity=True, lexicon=True),
    "s3": Stages(quantity=True, lexicon=True, spelling=True),
    "s4": Stages(quantity=True, lexicon=True, spelling=True, inflection=True),
    "s5": Stages(quantity=True, lexicon=True, spelling=True, inflection=True, brands=True),
    "s6": Stages(
        quantity=True, lexicon=True, spelling=True, inflection=True, brands=True, ordering=True
    ),
}


class QueryParser:
    def __init__(
        self,
        retrieval_vocabulary: Collection[str],
        spelling_frequency: Mapping[str, int],
        lexicon: Lexicon,
        settings: QuerySettings,
        stages: Stages,
        brands: BrandGazetteer | None = None,
    ) -> None:
        """``retrieval_vocabulary`` is the index's terms (plural forms must exist there);
        ``spelling_frequency`` is the spelling target vocabulary with document counts;
        ``brands`` is required when the brands stage is enabled."""
        if stages.brands and brands is None:
            raise ValueError("the brands stage needs a BrandGazetteer")
        self._vocabulary = frozenset(retrieval_vocabulary)
        self._lexicon = lexicon
        self._settings = settings
        self._stages = stages
        self._brands = brands if stages.brands else None
        protected = set(lexicon.single_terms)
        preferred: frozenset[str] = frozenset()
        if self._brands is not None:
            protected |= self._brands.protected_terms
            preferred = self._brands.brand_terms
        self._speller = (
            SpellingCorrector(
                spelling_frequency, protected, settings, preferred, phonetic=stages.brands
            )
            if stages.spelling
            else None
        )

    @property
    def speller(self) -> SpellingCorrector | None:
        return self._speller

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
        found = self._brands.search(tokens) if self._brands else None
        brand_at = {match.start: match for match in found.matches} if found else {}
        brand_matches = list(found.matches) if found else []
        notes = list(found.notes) if found else []
        consumed: set[int] = set()  # positions settled by a brand or a lexicon phrase
        resolved = list(tokens)  # tokens with spelling corrections applied
        text_at: dict[int, str] = {}  # token position -> text of the concept it formed
        position = 0
        while position < len(tokens):
            brand = brand_at.get(position)
            if brand is not None:
                formed = _brand_concepts(tokens[brand.start : brand.end], brand)
                concepts += formed
                for offset in range(brand.end - brand.start):
                    concept = formed[offset] if len(formed) > 1 else formed[0]
                    text_at[brand.start + offset] = concept.text
                consumed.update(range(brand.start, brand.end))
                position = brand.end
                continue
            match = self._lexicon.longest_match(tokens, position) if self._stages.lexicon else None
            if match is not None:
                length, entry = match
                phrase = tokens[position : position + length]
                concepts.append(_lexicon_concept(" ".join(phrase), [tuple(phrase)], entry))
                text_at.update(dict.fromkeys(range(position, position + length), concepts[-1].text))
                consumed.update(range(position, position + length))
                position += length
                continue
            token = tokens[position]
            before = len(corrections)
            concepts.append(self._token_concept(token, corrections))
            text_at[position] = concepts[-1].text
            if len(corrections) > before:
                resolved[position] = corrections[-1].corrected
            position += 1

        if self._brands is not None and resolved != tokens:
            # Brands typed with a spelling mistake ("amull") are recognised once corrected.
            again = self._brands.search(resolved, skip=consumed)
            brand_matches += [m for m in again.matches if m.start not in brand_at]
            notes += again.notes
        brand_matches.sort(key=lambda m: m.start)
        notes += [f"also names brand {m.display!r}; not used" for m in brand_matches[1:]]
        chosen = brand_matches[0] if brand_matches else None
        brand_concepts = (
            tuple(dict.fromkeys(text_at[i] for i in range(chosen.start, chosen.end)))
            if chosen
            else ()
        )

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
            brand=chosen.display if chosen else None,
            brand_key=chosen.key if chosen else None,
            brand_notes=tuple(dict.fromkeys(notes)),
            brand_concepts=brand_concepts,
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
            ((frozenset({token}),), (frozenset({correction.corrected}),)),
        )

    def _with_variants(self, concept: QueryConcept) -> QueryConcept:
        terms = list(concept.terms)
        present = {term.term for term in terms}
        for term in concept.terms:
            for form in variants(term.term, self._vocabulary):
                if form not in present:
                    terms.append(ConceptTerm(form, TermSource.INFLECTION, term.weight))
                    present.add(form)
        # A plural form joins its own word's slot: ("mustard", "oil") -> ("mustard", "oil|oils").
        alternatives = tuple(
            tuple(
                frozenset(
                    slot | {form for word in slot for form in variants(word, self._vocabulary)}
                )
                for slot in alternative
            )
            for alternative in concept.phrases
        )
        return QueryConcept(concept.text, tuple(terms), alternatives)


def _lexicon_concept(
    text: str,
    typed: list[tuple[str, ...]],
    entry: LexiconEntry,
    second_source: TermSource = TermSource.QUERY,
) -> QueryConcept:
    """Concept for a lexicon match. ``typed`` holds the shopper's phrase first, then
    (for a spelling correction) the corrected lexicon term."""
    groups = [(typed[0], TermSource.QUERY)]
    groups += [(extra, second_source) for extra in typed[1:]]
    groups += [(phrase, TermSource.LEXICON) for phrase in entry.english]
    return _weighted_concept(text, groups, drop_connectors=True)


def _brand_concepts(span: list[str], brand: BrandMatch) -> list[QueryConcept]:
    """Concepts for a recognised brand. Typed as the catalog spells it (ignoring a possessive
    "s"), each word
    stays its own concept (as without brand recognition). An alias or variant becomes one
    concept: the typed words plus the catalog spelling, each alternative weighing 1 in
    total ("coke" -> coke 1, coca 0.5, cola 0.5)."""
    if tuple(span) == brand.brand_words:
        return [QueryConcept.single(token) for token in span]
    return [
        _weighted_concept(
            " ".join(span),
            [(tuple(span), TermSource.QUERY), (brand.brand_words, TermSource.BRAND)],
        )
    ]


def _weighted_concept(
    text: str,
    groups: list[tuple[tuple[str, ...], TermSource]],
    *,
    drop_connectors: bool = False,
) -> QueryConcept:
    """One concept from alternative phrases; each phrase carries total weight 1, split
    evenly over its words (scoring), and becomes one alternative of ordered word slots
    (exact coverage). A word in several phrases keeps its highest weight and the source of
    its first appearance."""
    weights: dict[str, tuple[float, TermSource]] = {}
    alternatives: dict[tuple[str, ...], None] = {}  # ordered, without duplicates
    for phrase, source in groups:
        words = [w for w in phrase if not (drop_connectors and w in CONNECTORS)] or list(phrase)
        alternatives[tuple(words)] = None
        for word in words:
            weight = 1 / len(words)
            current = weights.get(word)
            if current is None or weight > current[0]:
                weights[word] = (weight, current[1] if current else source)
    return QueryConcept(
        text,
        tuple(ConceptTerm(word, source, weight) for word, (weight, source) in weights.items()),
        tuple(tuple(frozenset({word}) for word in words) for words in alternatives),
    )
