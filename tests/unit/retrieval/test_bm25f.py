"""BM25F behaviour against hand-calculated values.

Formula (see cartiq.retrieval.bm25f), k1 = 1.2 throughout:
    B_f = 1 - b + b * len / avglen;  tf~ = sum_f w_f * tf_f / B_f
    idf = ln(1 + (N - df + 0.5) / (df + 0.5));  score = sum_t idf * tf~ / (k1 + tf~)
Useful constants: ln(1.6) = 0.4700036292 (df = 2 of N = 3), ln(2) = 0.6931471806.
"""

import math
import random
from collections.abc import Collection, Mapping, Sequence

import pytest

from cartiq.config import BM25Settings, FieldWeight
from cartiq.domain import Product
from cartiq.retrieval import BM25FIndex, Field

OFF = FieldWeight(weight=0, b=0)


def settings(**fields: FieldWeight) -> BM25Settings:
    """Only the named fields are indexed."""
    return BM25Settings(k1=1.2, **{f.value: fields.get(f.value, OFF) for f in Field})


def index(docs: Mapping[str, Mapping[Field, Sequence[str]]], config: BM25Settings) -> BM25FIndex:
    return BM25FIndex.from_documents(list(docs), list(docs.values()), config)


def scores(idx: BM25FIndex, query: str, allowed: Collection[str] | None = None) -> dict[str, float]:
    return {c.product_id: c.bm25_score or 0.0 for c in idx.search(query, 100, allowed)}


NAME_ONLY = settings(name=FieldWeight(weight=1, b=0.75))
# d1 "a b" (len 2), d2 "a a c" (len 3), d3 "c" (len 1); avglen = 2.
CLASSIC = {
    "d1": {Field.NAME: ["a", "b"]},
    "d2": {Field.NAME: ["a", "a", "c"]},
    "d3": {Field.NAME: ["c"]},
}


class TestSingleField:
    def test_idf(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)

        assert idx.idf("a") == pytest.approx(0.4700036292)  # df 2: ln(1 + 1.5 / 2.5)
        assert idx.idf("b") == pytest.approx(0.9808292530)  # df 1: ln(1 + 2.5 / 1.5)
        assert idx.idf("zzz") is None

    def test_scores_match_hand_calculation(self) -> None:
        result = scores(index(CLASSIC, NAME_ONLY), "a")

        # d1: B = 1, tf~ = 1 -> 0.4700036 * 1 / 2.2
        assert result["d1"] == pytest.approx(0.2136380133)
        # d2: B = 0.25 + 0.75 * 3/2 = 1.375, tf~ = 2 / 1.375 = 1.4545 -> * 1.4545 / 2.6545
        assert result["d2"] == pytest.approx(0.2575362352)
        assert "d3" not in result

    def test_equals_classic_bm25_without_the_k1_plus_one_factor(self) -> None:
        def classic(tf: int, length: int, df: int, n: int = 3, avglen: float = 2.0) -> float:
            k1, b = 1.2, 0.75
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avglen))

        result = scores(index(CLASSIC, NAME_ONLY), "a")

        assert result["d1"] * 2.2 == pytest.approx(classic(tf=1, length=2, df=2))
        assert result["d2"] * 2.2 == pytest.approx(classic(tf=2, length=3, df=2))

    def test_query_terms_add_up(self) -> None:
        result = scores(index(CLASSIC, NAME_ONLY), "a b")

        assert result["d1"] == pytest.approx(0.2136380133 + 0.9808292530 / 2.2)


class TestSaturation:
    def test_term_frequency_saturates(self) -> None:
        docs = {f"d{tf}": {Field.NAME: ["t"] * tf} for tf in (1, 2, 4)} | {
            "other": {Field.NAME: ["x"]}
        }
        idx = index(docs, settings(name=FieldWeight(weight=1, b=0)))
        idf = idx.idf("t")
        assert idf is not None

        result = scores(idx, "t")

        for tf in (1, 2, 4):
            assert result[f"d{tf}"] == pytest.approx(idf * tf / (1.2 + tf))
        # Doubling tf adds less each time and never reaches idf.
        assert result["d2"] - result["d1"] > result["d4"] - result["d2"] > 0
        assert result["d4"] < idf

    def test_a_term_in_several_fields_saturates_once(self) -> None:
        config = settings(name=FieldWeight(weight=3, b=0.75), brand=FieldWeight(weight=1, b=0))
        docs = {
            "P1": {Field.NAME: ["x"], Field.BRAND: ["x"]},
            "P2": {Field.NAME: ["y"], Field.BRAND: ["z"]},
        }

        result = scores(index(docs, config), "x")

        # tf~ = 3 * 1 / 1 + 1 * 1 / 1 = 4 -> ln(2) * 4 / 5.2, not ln(2) * (3/4.2 + 1/2.2)
        assert result["P1"] == pytest.approx(0.6931471806 * 4 / 5.2)


LENGTHS = {
    "long": {Field.NAME: ["t", "x", "x"]},  # len 3
    "short": {Field.NAME: ["t"]},  # len 1; avglen = 2
    "other": {Field.NAME: ["y", "y"]},
}


class TestFieldNormalization:
    def test_b_zero_ignores_length(self) -> None:
        result = scores(index(LENGTHS, settings(name=FieldWeight(weight=1, b=0))), "t")

        assert result["long"] == result["short"]

    def test_b_favours_shorter_fields(self) -> None:
        idx = index(LENGTHS, settings(name=FieldWeight(weight=1, b=0.75)))
        idf = idx.idf("t")
        assert idf is not None

        result = scores(idx, "t")

        short_tf = 1 / (0.25 + 0.75 * 1 / 2)  # 1.6
        long_tf = 1 / (0.25 + 0.75 * 3 / 2)  # 0.7273
        assert result["short"] == pytest.approx(idf * short_tf / (1.2 + short_tf))
        assert result["long"] == pytest.approx(idf * long_tf / (1.2 + long_tf))

    def test_average_length_counts_only_documents_with_the_field(self) -> None:
        docs = {"a": {Field.NAME: ["x", "y", "z"]}, "b": {Field.NAME: ["x"]}, "c": {}}

        assert index(docs, NAME_ONLY).average_length(Field.NAME) == pytest.approx(2.0)


# name w=3 b=0.75 (avglen 4/3), brand w=1 b=0 (avglen 1).
TWO_FIELDS = settings(name=FieldWeight(weight=3, b=0.75), brand=FieldWeight(weight=1, b=0))
FORTUNE = {
    "P1": {Field.NAME: ["mustard", "oil"], Field.BRAND: ["fortune"]},
    "P2": {Field.NAME: ["atta"], Field.BRAND: ["fortune"]},
    "P3": {Field.NAME: ["oil"]},
}


class TestMultipleFields:
    def test_weighted_fields_match_hand_calculation(self) -> None:
        result = scores(index(FORTUNE, TWO_FIELDS), "fortune oil")

        # P1 oil: B = 0.25 + 0.75 * 2 / (4/3) = 1.375, tf~ = 3 / 1.375 = 2.1818
        #   -> 0.4700036 * 2.1818 / 3.3818 = 0.303228; fortune: 0.4700036 / 2.2 = 0.213638
        assert result["P1"] == pytest.approx(0.3032281479 + 0.2136380133)
        assert result["P2"] == pytest.approx(0.2136380133)
        # P3 oil: B = 0.25 + 0.75 * 1 / (4/3) = 0.8125, tf~ = 3.6923 -> * 3.6923 / 4.8923
        assert result["P3"] == pytest.approx(0.3547197202)

    def test_brand_only_match_ranks_below_product_match(self) -> None:
        ranked = [c.product_id for c in index(FORTUNE, TWO_FIELDS).search("fortune oil", 10)]

        assert ranked == ["P1", "P3", "P2"]


class TestZeroWeightFields:
    def test_do_not_affect_scores_df_or_lengths(self) -> None:
        with_category = {
            "P1": {Field.NAME: ["oil"], Field.CATEGORY: ["foodgrains", "oil"]},
            "P2": {Field.NAME: ["atta"], Field.CATEGORY: ["foodgrains", "oil", "masala"]},
            "P3": {Field.NAME: ["oil", "hair"]},
        }
        without = {pid: {Field.NAME: doc[Field.NAME]} for pid, doc in with_category.items()}

        a, b = index(with_category, NAME_ONLY), index(without, NAME_ONLY)

        assert a.idf("oil") == b.idf("oil")
        assert a.idf("foodgrains") is None
        assert a.average_length(Field.CATEGORY) is None
        assert a.average_length(Field.NAME) == b.average_length(Field.NAME)
        assert scores(a, "oil") == scores(b, "oil")
        assert "P2" not in scores(a, "oil masala")


class TestAllowed:
    def test_restricts_eligibility_without_changing_scores(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)
        everything = scores(idx, "a c")

        filtered = scores(idx, "a c", allowed={"d1", "d3"})

        assert filtered == {pid: everything[pid] for pid in ("d1", "d3")}

    def test_filters_before_the_limit(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)
        best = idx.search("a", 1)[0].product_id

        top = idx.search("a", 1, allowed={"d1", "d3"} - {best})

        assert best == "d2"
        assert [c.product_id for c in top] == ["d1"]
        assert top[0].bm25_rank == 1

    def test_empty_or_unknown_allowed_sets(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)

        assert idx.search("a", 10, allowed=set()) == []
        assert idx.search("a", 10, allowed={"nope"}) == []


class TestQueries:
    @pytest.mark.parametrize("query", ["", "   ", "!!!", "zzz", "zzz qqq"])
    def test_empty_or_unknown_queries_return_nothing(self, query: str) -> None:
        assert index(CLASSIC, NAME_ONLY).search(query, 10) == []

    def test_unknown_terms_are_ignored(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)

        assert idx.search("a zzz", 10) == idx.search("a", 10)

    def test_duplicate_query_terms_count_once(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)

        assert idx.search("a a A", 10) == idx.search("a", 10)


class TestTiesAndDeterminism:
    def test_equal_scores_are_ordered_by_product_id(self) -> None:
        docs = {pid: {Field.NAME: ["t"]} for pid in ("c", "a", "b")} | {"z": {Field.NAME: ["u"]}}

        ranked = [c.product_id for c in index(docs, NAME_ONLY).search("t", 10)]

        assert ranked == ["a", "b", "c"]

    def test_input_order_does_not_matter(self, synthetic_products: tuple[Product, ...]) -> None:
        shuffled = list(synthetic_products)
        random.Random(7).shuffle(shuffled)
        config = BM25Settings()

        a, b = BM25FIndex.build(synthetic_products, config), BM25FIndex.build(shuffled, config)

        for query in ("milk", "fortune oil", "amul milk 1 litre", "masala"):
            assert a.search(query, 50) == b.search(query, 50)

    def test_ranks_are_consecutive(self) -> None:
        result = index(CLASSIC, NAME_ONLY).search("a c", 10)

        assert [c.bm25_rank for c in result] == [1, 2, 3]
        assert all(c.fused_score == c.bm25_score for c in result)


class TestLimits:
    def test_depth_caps_results(self) -> None:
        idx = index(CLASSIC, NAME_ONLY)

        assert len(idx.search("a c", 1)) == 1
        assert len(idx.search("a c", 100)) == 3  # fewer matches than depth

    def test_depth_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            index(CLASSIC, NAME_ONLY).search("a", 0)

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            BM25FIndex.from_documents(["a", "a"], [{}, {}], NAME_ONLY)


class TestBrandVersusProductOnSyntheticCatalog:
    """Default (pre-registered) weights on the synthetic catalog."""

    @pytest.fixture
    def idx(self, synthetic_products: tuple[Product, ...]) -> BM25FIndex:
        return BM25FIndex.build(synthetic_products, BM25Settings())

    def test_fortune_oil(self, idx: BM25FIndex) -> None:
        rank = {c.product_id: c.bm25_rank or 0 for c in idx.search("fortune oil", 100)}
        fortune_oils = ["syn-0046", "syn-0047", "syn-0050"]
        other_oils = ["syn-0048", "syn-0049", "syn-0051"]
        fortune_non_oils = ["syn-0026", "syn-0027", "syn-0035", "syn-0040"]

        assert max(rank[p] for p in fortune_oils) < min(rank[p] for p in other_oils)
        assert max(rank[p] for p in other_oils) < min(rank[p] for p in fortune_non_oils)

    def test_mother_dairy_paneer(self, idx: BM25FIndex) -> None:
        rank = {c.product_id: c.bm25_rank or 0 for c in idx.search("mother dairy paneer", 100)}
        mother_dairy_non_paneer = ["syn-0005", "syn-0006", "syn-0007", "syn-0012"]

        assert rank["syn-0016"] == 1
        assert max(rank["syn-0014"], rank["syn-0015"]) < min(
            rank[p] for p in mother_dairy_non_paneer
        )

    def test_top_level_category_does_not_match(self, idx: BM25FIndex) -> None:
        # "Foodgrains, Oil & Masala" is not indexed, so atta does not match "oil".
        assert "syn-0023" not in {c.product_id for c in idx.search("oil", 100)}
