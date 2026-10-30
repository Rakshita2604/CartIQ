"""Versioned, reproducible data snapshots.

A snapshot is a directory ``<artifacts>/snapshots/<snapshot_id>/`` holding ``cartiq.sqlite``
and ``manifest.json``. The id is derived from a SHA-256 over the canonical content read back
from the database (plus the schema version), so identical inputs always produce the same id
and any drift is detectable with ``verify_snapshot``. ``created_at`` is recorded in the manifest
but is not part of the id.
"""

import hashlib
import json
import platform
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from cartiq import __version__
from cartiq.clock import Clock
from cartiq.config import SimulationSettings
from cartiq.ingestion.normalize import normalize_catalog
from cartiq.ingestion.raw import RawCatalogRow
from cartiq.simulation import simulate
from cartiq.storage import Repositories
from cartiq.storage.sqlite import (
    SCHEMA_VERSION,
    connect,
    initialize_schema,
    schema_version,
    sqlite_repositories,
)

DB_FILENAME = "cartiq.sqlite"
MANIFEST_FILENAME = "manifest.json"
_ID_PREFIX = "snap-"
_ID_HEX_LENGTH = 12


class SourceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["synthetic", "bigbasket"]
    name: str
    sha256: str


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    content_sha256: str
    schema_version: int
    created_at: AwareDatetime
    cartiq_version: str
    python_version: str
    source: SourceInfo
    simulation: SimulationSettings
    ingestion_report: dict[str, object]
    counts: dict[str, int]


class EmptyCatalogError(ValueError):
    pass


class SnapshotVerificationError(ValueError):
    pass


def snapshots_root(artifacts_dir: Path) -> Path:
    return artifacts_dir / "snapshots"


def build_snapshot(
    rows: Iterable[RawCatalogRow],
    source: SourceInfo,
    simulation: SimulationSettings,
    artifacts_dir: Path,
    clock: Clock,
) -> SnapshotManifest:
    catalog = normalize_catalog(rows)
    if not catalog.products:
        raise EmptyCatalogError(f"no valid products in source; report: {catalog.report.to_dict()}")
    world = simulate(catalog.products, simulation)

    root = snapshots_root(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        conn = connect(staging / DB_FILENAME)
        try:
            initialize_schema(conn)
            repos = sqlite_repositories(conn)
            repos.catalog.add_many(catalog.products)
            repos.stores.add_many(world.stores)
            repos.stock.upsert_many(world.stock)
            repos.users.add_many(world.users)
            repos.orders.add_many(world.orders)
            content_sha256 = content_hash(repos, SCHEMA_VERSION)
            counts = _counts(repos)
        finally:
            conn.close()

        snapshot_id = f"{_ID_PREFIX}{content_sha256[:_ID_HEX_LENGTH]}"
        final = root / snapshot_id
        if final.exists():
            # Same id means identical content; keep the existing snapshot untouched.
            return load_manifest(final)

        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            content_sha256=content_sha256,
            schema_version=SCHEMA_VERSION,
            created_at=clock.now(),
            cartiq_version=__version__,
            python_version=platform.python_version(),
            source=source,
            simulation=simulation,
            ingestion_report=catalog.report.to_dict(),
            counts=counts,
        )
        (staging / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2) + "\n")
        staging.rename(final)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_manifest(snapshot_dir: Path) -> SnapshotManifest:
    return SnapshotManifest.model_validate_json((snapshot_dir / MANIFEST_FILENAME).read_text())


def verify_snapshot(snapshot_dir: Path) -> SnapshotManifest:
    """Recompute the content hash read-only and compare it with the manifest."""
    manifest = load_manifest(snapshot_dir)
    conn = connect(snapshot_dir / DB_FILENAME, read_only=True)
    try:
        actual = content_hash(sqlite_repositories(conn), schema_version(conn))
    finally:
        conn.close()
    if actual != manifest.content_sha256:
        raise SnapshotVerificationError(
            f"{manifest.snapshot_id}: content hash {actual} != manifest {manifest.content_sha256}"
        )
    return manifest


def content_hash(repos: Repositories, schema: int) -> str:
    """SHA-256 over every entity in canonical order and canonical JSON form."""
    digest = hashlib.sha256(f"schema:{schema}\n".encode())
    tables: dict[str, list[Any]] = {
        "products": list(repos.catalog.list_all()),
        "stores": list(repos.stores.list_all()),
        "stock": list(repos.stock.list_all()),
        "users": list(repos.users.list_all()),
        "orders": list(repos.orders.list_all()),
    }
    for table, entities in tables.items():
        digest.update(f"table:{table}\n".encode())
        for entity in entities:
            line = json.dumps(asdict(entity), sort_keys=True, default=_json_default)
            digest.update(line.encode() + b"\n")
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows: Iterable[RawCatalogRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(asdict(row), sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def _counts(repos: Repositories) -> dict[str, int]:
    stock = repos.stock.list_all()
    return {
        "products": repos.catalog.count(),
        "stores": len(repos.stores.list_all()),
        "stock_records": len(stock),
        "out_of_stock_records": sum(not record.in_stock for record in stock),
        "users": len(repos.users.list_all()),
        "orders": len(repos.orders.list_all()),
    }


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")
