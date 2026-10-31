from collections.abc import Callable
from decimal import Decimal

import pytest

from cartiq.config import BM25Settings, QuerySettings
from cartiq.domain import ConceptTerm, ParsedQuery, Product, QueryConcept, TermSource
from cartiq.query import STAGES
from cartiq.query.factory import build_parser
from cartiq.ranking import DEFAULT_COVERAGE_FIELDS, CoverageIndex
from cartiq.retrieval import BM25FIndex


def product(
    pid: str, name: str, path: tuple[str, ...], brand: str | None = None, description: str = ""
) -> Product:
    return Product(pid, name, path, Decimal(10), brand=brand, description=description)


def concept(text: str, *alternatives: tuple[set[str], ...]) -> QueryConcept:
    words = sorted({w for alt in alternatives for slot in alt for w in slot})
    return QueryConcept(
        text,
        tuple(ConceptTerm(w, TermSource.QUERY) for w in words),
        tuple(tuple(frozenset(slot) for slot in alt) for alt in alternatives),
    )


OILS = (
    product(
        "mustard",
        "Kachi Ghani Mustard Oil",
        ("Foodgrains, Oil & Masala", "Edible Oils & Ghee", "Mustard Oil"),
    ),
    product(
        "sunflower",
        "Sunlite Refined Sunflower Oil",
        ("Foodgrains, Oil & Masala", "Edible Oils & Ghee", "Sunflower Oil"),
    ),
    product(
        "saffola",
        "Gold Refined Cooking Oil",
        ("Foodgrains, Oil & Masala", "Edible Oils & Ghee", "Sunflower Oil"),
        brand="Saffola",
    ),
    product("ghee", "Pure Ghee", ("Foodgrains, Oil & Masala", "Edible Oils & Ghee", "Ghee")),
    product("sarson", "Sarson Tel Pouch", ("Misc", "Misc", "Misc")),
)
SARSON = concept("sarson ka tel", ({"sarson"}, {"tel"}), ({"mustard"}, {"oil", "oils"}))


def test_every_slot_of_one_alternative_is_required() -> None:
    covered = CoverageIndex(OILS, DEFAULT_COVERAGE_FIELDS).covered(
        [SARSON], [p.product_id for p in OILS]
    )

    assert covered["mustard"] == {"sarson ka tel"}  # mustard + oil
    assert covered["sunflower"] == frozenset()  # oil alone does not make "mustard oil"
    assert covered["ghee"] == frozenset()  # "oils" only on the shelf label
    assert covered["sarson"] == {"sarson ka tel"}  # the other alternative, sarson + tel


def test_product_type_counts_but_shelf_and_category_do_not() -> None:
    sunflower = concept("sunflower oil", ({"sunflower"},))
    flours = [
        product("sooji", "Sooji Rava", ("Foodgrains", "Atta, Flours & Sooji", "Sooji")),
        product("besan", "Besan", ("Foodgrains", "Atta, Flours & Sooji", "Besan")),
        product("atta", "Chakki Atta", ("Foodgrains", "Atta, Flours & Sooji", "Atta")),
    ]
    index = CoverageIndex([*OILS, *flours], DEFAULT_COVERAGE_FIELDS)

    assert index.covered([sunflower], ["saffola"])["saffola"] == {"sunflower oil"}  # via type
    atta = index.covered([QueryConcept.single("atta")], ["sooji", "besan", "atta"])
    assert atta == {"sooji": frozenset(), "besan": frozenset(), "atta": {"atta"}}


def test_description_only_match_is_not_coverage() -> None:
    pan = product(
        "pan",
        "Non-Stick Frying Pan",
        ("Kitchen", "Cookware", "Fry Pans"),
        description="Perfect for eggs and omelettes",
    )
    index = CoverageIndex([pan], DEFAULT_COVERAGE_FIELDS)

    assert index.covered([QueryConcept.single("eggs")], ["pan"])["pan"] == frozenset()
    assert index.coverable([QueryConcept.single("eggs")]) == frozenset()


def test_multiword_alternatives_are_ored() -> None:
    thanda = concept(
        "thanda", ({"thanda"},), ({"cold"}, {"drink", "drinks"}), ({"soft"}, {"drink", "drinks"})
    )
    drinks = [
        product(
            "cola",
            "Soft Drink",
            ("Beverages", "Energy & Soft Drinks", "Cold Drinks"),
            brand="Coca-Cola",
        ),
        product("thums", "Thums Up 750 ml", ("Beverages", "Energy & Soft Drinks", "Cold Drinks")),
        product(
            "mango", "Mango Drink", ("Beverages", "Fruit Juices & Drinks", "Juices"), brand="Maaza"
        ),
    ]

    covered = CoverageIndex(drinks, DEFAULT_COVERAGE_FIELDS).covered(
        [thanda], ["cola", "thums", "mango"]
    )

    assert covered["cola"] == {"thanda"}  # soft drink (name) and cold drinks (type)
    assert covered["thums"] == {"thanda"}  # cold drinks (type)
    assert covered["mango"] == frozenset()  # "drink" alone is only half a phrase


def test_coverable_needs_one_product_with_every_slot() -> None:
    index = CoverageIndex(OILS, DEFAULT_COVERAGE_FIELDS)
    nothing_has_both = concept("x", ({"mustard"}, {"ghee"}))

    assert index.coverable([SARSON, nothing_has_both]) == {"sarson ka tel"}


def test_missing_fields_are_safe() -> None:
    loose = product("onion", "Fresh Onion", ("Vegetables",))  # no brand, no product type
    index = CoverageIndex([loose], DEFAULT_COVERAGE_FIELDS)

    assert index.covered([QueryConcept.single("onion")], ["onion"])["onion"] == {"onion"}


@pytest.mark.parametrize("alternatives", [((),), ((frozenset(),),)])
def test_empty_alternatives_or_slots_are_rejected(
    alternatives: tuple[tuple[frozenset[str], ...], ...],
) -> None:
    with pytest.raises(ValueError, match="non-empty word slots"):
        QueryConcept("x", (ConceptTerm("x", TermSource.QUERY),), alternatives)


def test_concept_without_alternatives_uses_one_per_term() -> None:
    plain = QueryConcept(
        "x", (ConceptTerm("a", TermSource.QUERY), ConceptTerm("b", TermSource.LEXICON))
    )

    assert plain.phrases == ((frozenset({"a"}),), (frozenset({"b"}),))


@pytest.fixture(scope="module")
def parse(synthetic_products: tuple[Product, ...]) -> Callable[[str], ParsedQuery]:
    index = BM25FIndex.build(synthetic_products, BM25Settings())
    parser = build_parser(list(synthetic_products), index.vocabulary, QuerySettings(), STAGES["s6"])
    return parser.parse


class TestParsedAlternatives:
    def test_hinglish_phrase_with_plural_slot(self, parse: Callable[[str], ParsedQuery]) -> None:
        (c,) = parse("sarson ka tel").concepts

        assert c.phrases == (
            (frozenset({"sarson"}), frozenset({"tel"})),
            (frozenset({"mustard"}), frozenset({"oil", "oils"})),
        )

    def test_spelling_alternatives(self, parse: Callable[[str], ParsedQuery]) -> None:
        (c,) = parse("detergant").concepts

        assert (frozenset({"detergent", "detergents"}),) in c.phrases
        assert (frozenset({"detergant"}),) in c.phrases

    def test_bm25_terms_are_unchanged_by_alternatives(
        self, parse: Callable[[str], ParsedQuery]
    ) -> None:
        (c,) = parse("sarson ka tel").concepts

        assert [(t.term, t.weight) for t in c.terms] == [
            ("sarson", 0.5),
            ("tel", 0.5),
            ("mustard", 0.5),
            ("oil", 0.5),
            ("oils", 0.5),
        ]


def test_construction_and_lookup_are_deterministic() -> None:
    import random

    shuffled = list(OILS)
    random.Random(1).shuffle(shuffled)
    a = CoverageIndex(OILS, DEFAULT_COVERAGE_FIELDS)
    b = CoverageIndex(shuffled, DEFAULT_COVERAGE_FIELDS)
    ids = [p.product_id for p in OILS]

    assert a.covered([SARSON], ids) == b.covered([SARSON], ids) == a.covered([SARSON], ids)
    assert a.coverable([SARSON]) == b.coverable([SARSON])
