from pathlib import Path

import numpy as np
import pytest

from cartiq.clock import FixedClock
from cartiq.domain import Product
from cartiq.embeddings.artifact import (
    EmbeddedCatalog,
    EmbeddingArtifactError,
    build_embeddings,
    load_embeddings,
    vectors_sha256,
)
from cartiq.embeddings.encoder import HashingEncoder
from cartiq.embeddings.templates import ProductTemplate


def build(products: tuple[Product, ...], directory: Path, clock: FixedClock) -> EmbeddedCatalog:
    return build_embeddings(
        products, HashingEncoder(), ProductTemplate.BASE, directory, clock, {"x": "1"}
    )


def test_build_records_identity_and_reloads(
    tmp_path: Path, synthetic_products: tuple[Product, ...], clock: FixedClock
) -> None:
    built = build(synthetic_products, tmp_path / "a", clock)

    assert built.manifest.encoder == "hashing-v1|dim64"
    assert built.manifest.product_template is ProductTemplate.BASE
    assert built.manifest.products == len(synthetic_products) == built.vectors.shape[0]
    assert list(built.product_ids) == sorted(p.product_id for p in synthetic_products)
    assert built.manifest.vectors_sha256 == vectors_sha256(built.vectors)
    assert not list(tmp_path.glob(".staging-*"))  # written atomically


def test_rebuild_is_byte_identical_and_shuffle_independent(
    tmp_path: Path, synthetic_products: tuple[Product, ...], clock: FixedClock
) -> None:
    import random

    shuffled = list(synthetic_products)
    random.Random(4).shuffle(shuffled)

    a = build(synthetic_products, tmp_path / "a", clock)
    b = build(tuple(shuffled), tmp_path / "b", clock)

    assert a.manifest.vectors_sha256 == b.manifest.vectors_sha256
    assert np.array_equal(a.vectors, b.vectors)


def test_refuses_another_catalog_or_altered_vectors(
    tmp_path: Path, synthetic_products: tuple[Product, ...], clock: FixedClock
) -> None:
    built = build(synthetic_products, tmp_path / "a", clock)

    with pytest.raises(EmbeddingArtifactError, match="built for catalog"):
        load_embeddings(tmp_path / "a", synthetic_products[:-1])
    np.save(tmp_path / "a" / "vectors.npy", built.vectors * 0.5)
    with pytest.raises(EmbeddingArtifactError, match="vectors sha256"):
        load_embeddings(tmp_path / "a", synthetic_products)


def test_refuses_to_reuse_vectors_from_another_encoder(
    tmp_path: Path, synthetic_products: tuple[Product, ...], clock: FixedClock
) -> None:
    build(synthetic_products, tmp_path / "a", clock)

    with pytest.raises(EmbeddingArtifactError, match="built with"):
        build_embeddings(
            synthetic_products, HashingEncoder(32), ProductTemplate.BASE, tmp_path / "a", clock, {}
        )
    with pytest.raises(EmbeddingArtifactError, match="built with"):
        build_embeddings(
            synthetic_products,
            HashingEncoder(),
            ProductTemplate.DESCRIPTION,
            tmp_path / "a",
            clock,
            {},
        )
