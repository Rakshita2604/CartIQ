from cartiq.config import BM25Settings
from cartiq.domain import Product
from cartiq.evaluation.pipelines import BM25Pipeline, TokenOverlapPipeline
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
