"""The embedded catalog: one versioned, verifiable artifact per (model, product template,
catalog).

``<artifacts>/embeddings/<model>-<revision12>-<template>-<catalog fingerprint12>/`` holds
``vectors.npy`` (float32, L2-normalised, rows in product-id order), ``ids.json`` and
``manifest.json`` (model identity and file hashes, template, pooling, normalisation,
maximum length, library versions, catalog fingerprint, vectors SHA-256). It is written to a
staging directory and renamed, and verified on load: a fingerprint or hash mismatch is
refused. Evaluations always read stored vectors, so results reproduce from the artifact.
"""

import hashlib
import json
import platform
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import AwareDatetime, BaseModel, ConfigDict

from cartiq.catalog_fingerprint import catalog_fingerprint
from cartiq.clock import Clock
from cartiq.domain import Product
from cartiq.embeddings.encoder import TextEncoder
from cartiq.embeddings.templates import ProductTemplate, product_text

VECTORS = "vectors.npy"
IDS = "ids.json"
MANIFEST = "manifest.json"


class EmbeddingArtifactError(RuntimeError):
    pass


class EmbeddingManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    encoder: str  # TextEncoder.identity: model, revision, file hashes, pooling, prefixes
    product_template: ProductTemplate
    catalog_sha256: str
    products: int
    dimension: int
    normalization: str
    vectors_sha256: str
    libraries: dict[str, str]
    created_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class EmbeddedCatalog:
    manifest: EmbeddingManifest
    product_ids: tuple[str, ...]
    vectors: npt.NDArray[np.float32]


def artifact_id(model_key: str, revision: str, template: ProductTemplate, fingerprint: str) -> str:
    return f"{model_key}-{revision[:12]}-{template.value}-{fingerprint[:12]}"


def vectors_sha256(vectors: npt.NDArray[np.float32]) -> str:
    digest = hashlib.sha256(f"{vectors.dtype}|{vectors.shape}\n".encode())
    digest.update(np.ascontiguousarray(vectors).tobytes())
    return digest.hexdigest()


def build_embeddings(
    products: Sequence[Product],
    encoder: TextEncoder,
    template: ProductTemplate,
    directory: Path,
    clock: Clock,
    libraries: dict[str, str],
) -> EmbeddedCatalog:
    """Encode the catalog into ``directory`` (atomically), or load it if already built.
    An existing artifact is reused only if it was built by this encoder and template."""
    if directory.exists():
        existing = load_embeddings(directory, products)
        built_with = (existing.manifest.encoder, existing.manifest.product_template)
        if built_with != (encoder.identity, template):
            raise EmbeddingArtifactError(
                f"{directory.name}: built with {built_with}, not ({encoder.identity!r}, "
                f"{template.value!r}); remove it to rebuild"
            )
        return existing
    ordered = sorted(products, key=lambda product: product.product_id)
    vectors = encoder.encode_documents([product_text(p, template) for p in ordered])
    manifest = EmbeddingManifest(
        artifact_id=directory.name,
        encoder=encoder.identity,
        product_template=template,
        catalog_sha256=catalog_fingerprint(ordered),
        products=len(ordered),
        dimension=int(vectors.shape[1]),
        normalization="l2",
        vectors_sha256=vectors_sha256(vectors),
        libraries={"python": platform.python_version(), "numpy": np.__version__, **libraries},
        created_at=clock.now(),
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=directory.parent))
    try:
        np.save(staging / VECTORS, vectors)
        (staging / IDS).write_text(json.dumps([p.product_id for p in ordered]) + "\n")
        (staging / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n")
        staging.rename(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return load_embeddings(directory, products)


def load_embeddings(directory: Path, products: Sequence[Product]) -> EmbeddedCatalog:
    """Load and verify: catalog fingerprint, ids and the vectors' hash must all match."""
    manifest = EmbeddingManifest.model_validate_json((directory / MANIFEST).read_text())
    ids = tuple(json.loads((directory / IDS).read_text()))
    vectors = np.load(directory / VECTORS)
    fingerprint = catalog_fingerprint(products)
    if manifest.catalog_sha256 != fingerprint:
        raise EmbeddingArtifactError(
            f"{directory.name}: built for catalog {manifest.catalog_sha256[:12]}, "
            f"not {fingerprint[:12]}"
        )
    if ids != tuple(sorted(p.product_id for p in products)) or len(ids) != vectors.shape[0]:
        raise EmbeddingArtifactError(f"{directory.name}: product ids do not match the catalog")
    actual = vectors_sha256(vectors)
    if actual != manifest.vectors_sha256:
        raise EmbeddingArtifactError(
            f"{directory.name}: vectors sha256 {actual} != manifest {manifest.vectors_sha256}"
        )
    return EmbeddedCatalog(manifest, ids, vectors)
