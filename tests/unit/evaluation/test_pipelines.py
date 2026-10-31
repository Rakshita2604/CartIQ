from collections.abc import Collection, Iterable

import pytest

from cartiq.config import BM25Settings
from cartiq.domain import Product
from cartiq.evaluation.pipelines import BM25Pipeline, TokenOverlapPipeline, UnderstandingPipeline
from cartiq.retrieval import BM25FIndex, bm25_config_id

IN_STOCK = {"S1": frozenset({"syn-0002", "syn-0004", "syn-0009"}), "S2": frozenset()}


class TestTokenOverlap:
    def test_retrieves_by_overlap_then_product_id(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        results = TokenOverlapPipeline(synthetic_products, IN_STOCK).retrieve("amul milk", depth=5)

        # Amul milks match both terms and come first, in id order.
        assert list(results[:4]) == ["syn-0001", "syn-0002", "syn-0003", "syn-0004"]
        assert len(results) == 5

    def test_search_keeps_only_products_in_stock(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        pipeline = TokenOverlapPipeline(synthetic_products, IN_STOCK)

        staged = pipeline.search("amul milk", "S1", limit=10)

        assert list(staged.final) == ["syn-0002", "syn-0004", "syn-0009"]
        assert staged.available == staged.final  # no fallbacks or substitutions
        assert not pipeline.search("amul milk", "S2", limit=10).final

    def test_no_overlap_returns_nothing(self, synthetic_products: tuple[Product, ...]) -> None:
        assert not TokenOverlapPipeline(synthetic_products, IN_STOCK).retrieve("doodh", depth=10)

    def test_matches_category_text(self, synthetic_products: tuple[Product, ...]) -> None:
        results = TokenOverlapPipeline(synthetic_products, IN_STOCK).retrieve("curd", depth=10)

        # "Masti Dahi" products are found through their category path, not their name.
        assert "syn-0010" in results


class TestBM25Pipeline:
    def test_name_carries_the_config_identity(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        config = BM25Settings()

        pipeline = BM25Pipeline(synthetic_products, IN_STOCK, config)

        assert pipeline.name == f"bm25f-{bm25_config_id(config)}"
        assert bm25_config_id(BM25Settings(k1=1.5)) != bm25_config_id(config)

    def test_retrieve_ignores_stock(self, synthetic_products: tuple[Product, ...]) -> None:
        index = BM25FIndex.build(synthetic_products, BM25Settings())
        pipeline = BM25Pipeline(synthetic_products, IN_STOCK, BM25Settings())

        assert list(pipeline.retrieve("milk", 100)) == [
            c.product_id for c in index.search("milk", 100)
        ]

    def test_search_is_store_filtered_global_order(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        pipeline = BM25Pipeline(synthetic_products, IN_STOCK, BM25Settings())
        global_order = list(pipeline.retrieve("amul milk", 100))

        staged = pipeline.search("amul milk", "S1", limit=10)

        assert list(staged.final) == [p for p in global_order if p in IN_STOCK["S1"]]
        assert staged.available == staged.final
        assert not pipeline.search("amul milk", "S2", limit=10).final
        assert not pipeline.search("amul milk", "unknown-store", limit=10).final


class TestUnderstandingPipeline:
    def test_all_stages_off_reproduces_plain_bm25(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        from pathlib import Path

        from cartiq.config import QuerySettings
        from cartiq.evaluation import load_judgments
        from cartiq.evaluation.pipelines import UnderstandingPipeline
        from cartiq.query import Stages

        bm25 = BM25Pipeline(synthetic_products, IN_STOCK, BM25Settings())
        off = UnderstandingPipeline(
            synthetic_products, IN_STOCK, BM25Settings(), QuerySettings(), "off", Stages()
        )
        queries = [q.text for q in load_judgments(Path("eval/judgments/synthetic-v1.toml")).queries]

        for query in queries:
            assert off.retrieve(query, 100) == bm25.retrieve(query, 100), query
            assert off.search(query, "S1", 10) == bm25.search(query, "S1", 10), query

    def test_name_identifies_stage_and_configuration(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        from cartiq.config import QuerySettings
        from cartiq.evaluation.pipelines import UnderstandingPipeline
        from cartiq.query import STAGES

        def name(stage: str, settings: QuerySettings) -> str:
            return UnderstandingPipeline(
                synthetic_products, IN_STOCK, BM25Settings(), settings, stage, STAGES[stage]
            ).name

        assert name("s2", QuerySettings()).startswith("qu-s2-")
        assert name("s2", QuerySettings()) != name("s3", QuerySettings())
        assert name("s3", QuerySettings()) != name("s3", QuerySettings(dominance_ratio=10))


def test_stages_without_spelling_keep_their_phase_4a_identity(
    synthetic_products: tuple[Product, ...],
) -> None:
    """Settings added in Phase 4b (spelling_fields) must not change S1/S2 identities."""
    from cartiq.config import QuerySettings
    from cartiq.evaluation.pipelines import UnderstandingPipeline
    from cartiq.query import STAGES

    def name(stage: str) -> str:
        return UnderstandingPipeline(
            synthetic_products, IN_STOCK, BM25Settings(), QuerySettings(), stage, STAGES[stage]
        ).name

    assert name("s1") == "qu-s1-fd4ccaebd607"
    assert name("s2") == "qu-s2-dec5aa12f51c"
    assert name("s3") != "qu-s3-7fc6ed0f33da"  # spelling vocabulary changed in 4b
    # Brand settings (4b, stage s5) must not change the reviewed S3/S4 identities.
    assert name("s3") == "qu-s3-3d991413c1db"
    assert name("s4") == "qu-s4-5aa9298020a9"
    assert name("s5").startswith("qu-s5-")


class TestConstraintOrdering:
    """Stage s6 on hand-built catalogs."""

    @staticmethod
    def pipeline(
        products: list[Product], in_stock: dict[str, frozenset[str]], depth: int = 100
    ) -> UnderstandingPipeline:
        from cartiq.config import QuerySettings, RankingSettings
        from cartiq.evaluation.pipelines import UnderstandingPipeline
        from cartiq.query import STAGES

        return UnderstandingPipeline(
            products,
            in_stock,
            BM25Settings(),
            QuerySettings(),
            "s6",
            STAGES["s6"],
            RankingSettings(candidate_depth=depth),
        )

    @staticmethod
    def milk(pid: str, brand: str, size: str, words: str = "Toned Milk") -> Product:
        from decimal import Decimal

        from cartiq.parsing.quantity import parse_quantity

        parsed = parse_quantity(size)
        assert parsed is not None
        return Product(
            pid,
            f"{brand} {words} {size}",
            ("Dairy", "Dairy", "Milk"),
            Decimal(10),
            brand=brand,
            quantity=parsed.quantity,
            pack_count=parsed.pack_count,
        )

    QUERY = "milk 1 litre"

    def catalog(self) -> list[Product]:
        # Twelve short-named 500 ml milks outrank the one long-named 1 l milk by BM25F
        # (length normalisation), so the EXACT product starts below rank 10.
        short = [self.milk(f"n{i:02d}", "Nandini", "500 ml", "Milk") for i in range(12)]
        exact = self.milk("one", "Amul", "1 l", "Taaza Homogenised Toned Farm Fresh Milk Pouch")
        return [*short, exact]

    def test_exact_below_rank_ten_is_promoted(self) -> None:
        stock = {"S": frozenset(p.product_id for p in self.catalog())}
        pipe = self.pipeline(self.catalog(), stock)

        plain = pipe.retrieve(self.QUERY, 100)
        staged = pipe.search(self.QUERY, "S", 10)
        first = pipe.ordered(self.QUERY).scored[0]

        assert plain.index("one") >= 10  # BM25F rank 11 or lower
        assert first.product_id == "one"
        assert first.features["bm25_rank"] > 10  # original rank kept
        assert next(iter(staged.final)) == "one"
        assert list(staged.available) == ["one"]

    def test_store_eligibility_applies_before_the_depth_cut(self) -> None:
        # Depth 3: globally the 1 l milk is not in the top 3, but this store stocks it.
        pipe = self.pipeline(self.catalog(), {"S": frozenset({"one", "n00"})}, depth=3)

        staged = pipe.search(self.QUERY, "S", 10)

        assert list(staged.final) == ["one", "n00"]

    def test_out_of_stock_exact_leaves_relaxed_results(self) -> None:
        pipe = self.pipeline(self.catalog(), {"S": frozenset({"n00", "n01"})})

        staged = pipe.search(self.QUERY, "S", 10)

        assert list(staged.available) == []  # available-zero (no in-stock EXACT) ...
        assert list(staged.final) == ["n00", "n01"]  # ... but not final-zero

    def test_empty_store(self) -> None:
        pipe = self.pipeline(self.catalog(), {"S": frozenset()})

        staged = pipe.search(self.QUERY, "S", 10)

        assert (list(staged.available), list(staged.final)) == ([], [])

    def test_shuffled_input_gives_identical_sections(self) -> None:
        import random

        products = self.catalog()
        shuffled = list(products)
        random.Random(3).shuffle(shuffled)
        stock = {"S": frozenset(p.product_id for p in products)}

        a = self.pipeline(products, stock).ordered(self.QUERY)
        b = self.pipeline(shuffled, stock).ordered(self.QUERY)

        assert a.scored == b.scored

    def test_labels_its_available_zero_definition(self) -> None:
        pipe = self.pipeline(self.catalog(), {})

        assert pipe.available_zero_definition is not None
        assert pipe.available_zero_definition.startswith("in-stock EXACT products (v2)")

    def test_s5_and_earlier_have_no_definition_and_keep_their_identity(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        from cartiq.config import QuerySettings
        from cartiq.evaluation.pipelines import UnderstandingPipeline
        from cartiq.query import STAGES

        s5 = UnderstandingPipeline(
            synthetic_products, IN_STOCK, BM25Settings(), QuerySettings(), "s5", STAGES["s5"]
        )

        assert s5.available_zero_definition is None
        assert s5.name == "qu-s5-969762a328bb"


class TestFinalAudit:
    """Phase 4b final audit: coverage-index lifecycle, BM25 preservation, S6 fingerprint."""

    def test_coverage_index_is_built_once_and_reused(
        self, synthetic_products: tuple[Product, ...], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cartiq.config import QuerySettings, RankingSettings
        from cartiq.evaluation import pipelines
        from cartiq.query import STAGES
        from cartiq.ranking import CoverageIndex

        built = []

        class Counting(CoverageIndex):
            def __init__(self, products: Iterable[Product], fields: Collection[str]) -> None:
                built.append(1)
                super().__init__(products, fields)

        monkeypatch.setattr(pipelines, "CoverageIndex", Counting)
        pipe = UnderstandingPipeline(
            synthetic_products,
            IN_STOCK,
            BM25Settings(),
            QuerySettings(),
            "s6",
            STAGES["s6"],
            RankingSettings(),
        )
        for query in ("milk", "amul milk 1 litre", "sarson ka tel", "coke 6 x 300 ml"):
            pipe.retrieve(query, 100)
            for store in IN_STOCK:
                pipe.search(query, store, 10)
            pipe.ordered(query)

        assert built == [1]

    def test_s6_keeps_bm25_scores_and_ranks(self, synthetic_products: tuple[Product, ...]) -> None:
        from cartiq.config import QuerySettings, RankingSettings
        from cartiq.query import STAGES

        pipe = UnderstandingPipeline(
            synthetic_products,
            IN_STOCK,
            BM25Settings(),
            QuerySettings(),
            "s6",
            STAGES["s6"],
            RankingSettings(),
        )
        for query in ("amul milk 1 litre", "fortune oil", "sarson ka tel", "eggs 12"):
            plain = pipe._index.search_concepts(pipe.parse(query).concepts, 100)
            by_id = {c.product_id: c for c in plain}
            ordered = pipe.ordered(query).scored

            assert {s.product_id for s in ordered} == set(by_id)
            for scored in ordered:
                assert scored.score == by_id[scored.product_id].bm25_score
                assert scored.features["bm25_rank"] == by_id[scored.product_id].bm25_rank

    def test_s6_fingerprint_is_frozen_and_tracks_its_settings(
        self, synthetic_products: tuple[Product, ...]
    ) -> None:
        from cartiq.config import QuerySettings, RankingSettings
        from cartiq.query import STAGES

        def name(
            stage: str,
            query: QuerySettings | None = None,
            ranking: RankingSettings | None = None,
            bm25: BM25Settings | None = None,
        ) -> str:
            return UnderstandingPipeline(
                synthetic_products,
                IN_STOCK,
                bm25 or BM25Settings(),
                query or QuerySettings(),
                stage,
                STAGES[stage],
                ranking or RankingSettings(),
            ).name

        assert name("s6") == "qu-s6-a9d842bd342f"  # Phase 4b frozen configuration
        changed = {
            name("s6", ranking=RankingSettings(candidate_depth=50)),
            name("s6", ranking=RankingSettings(coverage_fields=("name", "brand"))),
            name("s6", query=QuerySettings(brand_min_share=0.6)),
            name("s6", query=QuerySettings(dominance_ratio=10)),
            name("s6", bm25=BM25Settings(k1=1.5)),
        }
        assert "qu-s6-a9d842bd342f" not in changed
        assert len(changed) == 5
        # Ranking settings only exist for s6: earlier stages keep their identities.
        assert name("s5", ranking=RankingSettings(candidate_depth=50)) == "qu-s5-969762a328bb"
