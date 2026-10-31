from decimal import Decimal

from cartiq.config import BM25Settings, QuerySettings
from cartiq.domain import Product
from cartiq.embeddings.templates import ProductTemplate, QueryTemplate, product_text, query_text
from cartiq.query import STAGES
from cartiq.query.factory import build_parser
from cartiq.retrieval import BM25FIndex


def product(name: str, brand: str | None, path: tuple[str, ...], description: str = "") -> Product:
    return Product("P1", name, path, Decimal(10), brand=brand, description=description)


def test_base_template_does_not_repeat_the_brand() -> None:
    assert product_text(
        product("Amul Taaza Milk 1 l", "Amul", ("Dairy", "Dairy", "Milk")), ProductTemplate.BASE
    ) == ("Amul Taaza Milk 1 l. Milk.")
    assert product_text(
        product("Homogenized Milk", "Amul", ("Dairy", "Dairy", "Milk")), ProductTemplate.BASE
    ) == ("Amul Homogenized Milk. Milk.")


def test_missing_brand_and_product_type() -> None:
    assert (
        product_text(product("Fresh Onion", None, ("Vegetables",)), ProductTemplate.BASE)
        == "Fresh Onion."
    )


def test_description_template_adds_the_first_32_words() -> None:
    words = " ".join(f"w{i}" for i in range(40))
    text = product_text(
        product("Ghee", "Amul", ("Dairy", "Ghee"), words), ProductTemplate.DESCRIPTION
    )

    assert text == "Amul Ghee. Ghee. " + " ".join(f"w{i}" for i in range(32))
    assert (
        product_text(product("Ghee", "Amul", ("Dairy", "Ghee")), ProductTemplate.DESCRIPTION)
        == "Amul Ghee. Ghee."
    )


def test_query_templates(synthetic_products: tuple[Product, ...]) -> None:
    index = BM25FIndex.build(synthetic_products, BM25Settings())
    parse = build_parser(
        list(synthetic_products), index.vocabulary, QuerySettings(), STAGES["s6"]
    ).parse

    assert query_text("amul milk 1 litre", parse, QueryTemplate.RAW) == "amul milk"
    assert query_text("12", parse, QueryTemplate.RAW) == "12"  # nothing but a quantity: keep it
    assert query_text("doodh", parse, QueryTemplate.EXPANDED) == "doodh milk"
    assert query_text("amull milk", parse, QueryTemplate.EXPANDED) == "amull milk amul"
    assert query_text("coke 6 x 300 ml", parse, QueryTemplate.EXPANDED) == "coke coca cola"
    # Plural forms are not added; lexicon English is.
    assert query_text("sarson ka tel", parse, QueryTemplate.EXPANDED) == "sarson ka tel mustard oil"
