import pytest

from cartiq.domain import Quantity, QuantityKind, Unit
from cartiq.query.quantities import extract_quantity


def extract(text: str):  # type: ignore[no-untyped-def]
    return extract_quantity(text, bare_count_max=50)


@pytest.mark.parametrize(
    ("text", "kind", "quantity", "pack_count", "retrieval_text"),
    [
        ("amul milk 1 litre", QuantityKind.AMOUNT, Quantity(1, Unit.L), None, "amul milk"),
        ("milk 500 ml", QuantityKind.AMOUNT, Quantity(500, Unit.ML), None, "milk"),
        ("atta 10kg", QuantityKind.AMOUNT, Quantity(10, Unit.KG), None, "atta"),
        ("coke 6 x 300 ml", QuantityKind.AMOUNT, Quantity(300, Unit.ML), 6, "coke"),
        ("eggs 12 pieces", QuantityKind.AMOUNT, Quantity(12, Unit.PCS), None, "eggs"),
        ("maggi 12 pack", QuantityKind.PACK, None, 12, "maggi"),
        ("biscuits 3-pack", QuantityKind.PACK, None, 3, "biscuits"),
        ("soap pack of 6", QuantityKind.PACK, None, 6, "soap"),
        ("300 ml coke 6 pack", QuantityKind.AMOUNT, Quantity(300, Unit.ML), 6, "coke"),
        ("1 dozen eggs", QuantityKind.AMOUNT, Quantity(12, Unit.PCS), None, "eggs"),
        ("dozen bananas", QuantityKind.AMOUNT, Quantity(12, Unit.PCS), None, "bananas"),
        ("half dozen bananas", QuantityKind.AMOUNT, Quantity(6, Unit.PCS), None, "bananas"),
        ("do kilo aata", QuantityKind.AMOUNT, Quantity(2, Unit.KG), None, "aata"),
        ("aadha kilo paneer", QuantityKind.AMOUNT, Quantity(0.5, Unit.KG), None, "paneer"),
        ("ek litre doodh", QuantityKind.AMOUNT, Quantity(1, Unit.L), None, "doodh"),
    ],
)
def test_recognised_quantities(
    text: str,
    kind: QuantityKind,
    quantity: Quantity | None,
    pack_count: int | None,
    retrieval_text: str,
) -> None:
    result = extract(text)

    assert result.mention is not None
    assert result.mention.kind is kind
    assert result.mention.quantity == quantity
    assert result.mention.pack_count == pack_count
    assert result.retrieval_text == retrieval_text


class TestTrailingNumbers:
    def test_bare_trailing_integer_is_an_ambiguous_count_not_a_pack(self) -> None:
        mention = extract("eggs 12").mention

        assert mention is not None
        assert mention.kind is QuantityKind.BARE_NUMBER
        assert (mention.count, mention.pack_count, mention.quantity) == (12, None, None)

    def test_number_that_is_part_of_a_product_name_is_still_only_ambiguous(self) -> None:
        # "Lifebuoy Total 10" is a product name; the reading stays explicitly ambiguous.
        mention = extract("lifebuoy total 10").mention

        assert mention is not None
        assert mention.kind is QuantityKind.BARE_NUMBER

    @pytest.mark.parametrize(
        "text",
        [
            "dettol 3 in 1",  # 1 is never a count
            "colgate 360",  # above bare_count_max
            "12",  # nothing but the number
            "eggs 1",
        ],
    )
    def test_trailing_numbers_left_alone(self, text: str) -> None:
        result = extract(text)

        assert result.mention is None
        assert result.retrieval_text == text


@pytest.mark.parametrize(
    "text",
    [
        "maggi 2 minute noodles",  # number inside a product name
        "100% pure coconut oil",
        "7up",
        "what to do",  # "do" is a number word only before a unit
        "char minar biscuits",
        "milk 500 ml 1 l",  # two different sizes: ambiguous
        "diapers 12-17 kg",  # a range is not a quantity
        "",
    ],
)
def test_no_quantity(text: str) -> None:
    result = extract(text)

    assert result.mention is None
    assert result.retrieval_text == " ".join(text.split())


class TestSpans:
    def test_spans_index_the_original_text(self) -> None:
        text = "Amul Milk 1 LITRE"
        mention = extract(text).mention

        assert mention is not None
        assert mention.spans == ((10, 17),)
        assert text[10:17] == mention.text == "1 LITRE"

    def test_number_word_span_covers_the_word(self) -> None:
        mention = extract("do kilo aata").mention

        assert mention is not None
        assert mention.text == "do kilo"

    def test_two_spans_keep_the_words_between_them(self) -> None:
        result = extract("300 ml thums up 6 pack")

        assert result.mention is not None
        assert result.mention.spans == ((0, 6), (16, 22))
        assert result.retrieval_text == "thums up"
