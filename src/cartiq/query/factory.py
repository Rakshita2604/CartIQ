"""Build a QueryParser for a catalog, and the identity of that configuration."""

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import asdict
from typing import Any

from cartiq.config import QuerySettings
from cartiq.domain import Product
from cartiq.query.brands import BrandGazetteer, WordStatistics, add_word_statistics
from cartiq.query.lexicon import load_brand_aliases, load_lexicon
from cartiq.query.parser import PARSER_VERSION, QueryParser, Stages
from cartiq.retrieval.documents import Field, product_fields


def build_parser(
    products: Sequence[Product],
    retrieval_vocabulary: Collection[str],
    settings: QuerySettings,
    stages: Stages,
) -> QueryParser:
    lexicon = load_lexicon(settings.lexicon)
    # One streaming pass: each product is tokenized once and not kept.
    spelling_fields = [Field(name) for name in settings.spelling_fields]
    spelling_counts: Counter[str] = Counter()
    in_brand: Counter[str] = Counter()
    anywhere: Counter[str] = Counter()
    for product in products:
        document = product_fields(product)
        spelling_counts.update({token for field in spelling_fields for token in document[field]})
        if stages.brands:
            add_word_statistics(document, in_brand, anywhere)
    brands = None
    if stages.brands:
        aliases, _ = load_brand_aliases(settings.brand_aliases)
        brands = BrandGazetteer(
            (product.brand for product in products if product.brand),
            aliases,
            WordStatistics(in_brand=dict(in_brand), anywhere=dict(anywhere)),
            settings.brand_min_share,
        )
    spelling_df = dict(spelling_counts)
    return QueryParser(retrieval_vocabulary, spelling_df, lexicon, settings, stages, brands)


def parser_identity(settings: QuerySettings, stages: Stages) -> dict[str, Any]:
    """Everything besides the catalog that determines parser output.

    Settings and stage flags added after Phase 4a enter only when the stage that uses them
    is enabled, so earlier stages keep their earlier identities (and report file names).
    """
    flags = asdict(stages)
    for added in ("brands", "ordering"):  # stage flags added after Phase 4a
        if not flags[added]:
            del flags[added]
    identity: dict[str, Any] = {
        "parser": PARSER_VERSION,
        "lexicon": load_lexicon(settings.lexicon).sha256,
        "query": settings.model_dump(mode="json", exclude={"spelling_fields", "brand_min_share"}),
        "stages": flags,
    }
    if stages.spelling:
        identity["spelling_fields"] = list(settings.spelling_fields)
    if stages.brands:
        # Rule versions within the brands stage (4b review): Soundex tie-break, brand-share guard.
        identity["brand_rules"] = "soundex-tiebreak+brand-share-v2"
        identity["brand_min_share"] = settings.brand_min_share
        identity["brand_aliases"] = load_brand_aliases(settings.brand_aliases)[1]
    return identity
