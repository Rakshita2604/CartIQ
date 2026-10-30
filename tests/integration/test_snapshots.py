import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from cartiq.clock import FixedClock
from cartiq.config import SimulationSettings
from cartiq.ingestion.raw import RawCatalogRow
from cartiq.ingestion.synthetic import SYNTHETIC_CATALOG_VERSION, synthetic_catalog_rows
from cartiq.snapshots import (
    DB_FILENAME,
    EmptyCatalogError,
    SnapshotManifest,
    SnapshotVerificationError,
    SourceInfo,
    build_snapshot,
    load_manifest,
    rows_sha256,
    snapshots_root,
    verify_snapshot,
)
from cartiq.storage.sqlite import connect, sqlite_repositories


def _build(artifacts: Path, settings: SimulationSettings, clock: FixedClock) -> SnapshotManifest:
    rows = synthetic_catalog_rows()
    source = SourceInfo(kind="synthetic", name=SYNTHETIC_CATALOG_VERSION, sha256=rows_sha256(rows))
    return build_snapshot(rows, source, settings, artifacts, clock)


def test_build_writes_manifest_and_database(
    tmp_path: Path, small_simulation: SimulationSettings, clock: FixedClock
) -> None:
    manifest = _build(tmp_path, small_simulation, clock)
    snapshot_dir = snapshots_root(tmp_path) / manifest.snapshot_id

    assert manifest.snapshot_id == f"snap-{manifest.content_sha256[:12]}"
    assert manifest.created_at == clock.now()
    assert manifest.simulation == small_simulation
    assert load_manifest(snapshot_dir) == manifest
    assert manifest.counts["products"] == manifest.ingestion_report["accepted"]
    assert manifest.counts["stores"] == small_simulation.n_stores
    assert manifest.counts["users"] == small_simulation.n_users
    assert [p.name for p in snapshot_dir.parent.iterdir()] == [manifest.snapshot_id]


def test_rebuild_is_reproducible_and_keeps_original(
    tmp_path: Path, small_simulation: SimulationSettings, clock: FixedClock
) -> None:
    first = _build(tmp_path, small_simulation, clock)
    clock.advance(timedelta(days=1))
    second = _build(tmp_path, small_simulation, clock)
    elsewhere = _build(tmp_path / "other", small_simulation, clock)

    assert second == first  # existing snapshot returned unchanged, including created_at
    assert elsewhere.content_sha256 == first.content_sha256
    assert elsewhere.created_at != first.created_at


def test_changing_simulation_changes_snapshot_id(
    tmp_path: Path, small_simulation: SimulationSettings, clock: FixedClock
) -> None:
    first = _build(tmp_path, small_simulation, clock)
    second = _build(tmp_path, small_simulation.model_copy(update={"seed": 99}), clock)

    assert first.snapshot_id != second.snapshot_id
    assert len(list(snapshots_root(tmp_path).iterdir())) == 2


def test_verify_detects_tampering(
    tmp_path: Path, small_simulation: SimulationSettings, clock: FixedClock
) -> None:
    manifest = _build(tmp_path, small_simulation, clock)
    snapshot_dir = snapshots_root(tmp_path) / manifest.snapshot_id
    assert verify_snapshot(snapshot_dir) == manifest

    conn = sqlite3.connect(snapshot_dir / DB_FILENAME)
    with conn:
        conn.execute("UPDATE stock SET available_qty = available_qty + 1")
    conn.close()

    with pytest.raises(SnapshotVerificationError):
        verify_snapshot(snapshot_dir)


def test_snapshot_database_opens_read_only(
    tmp_path: Path, small_simulation: SimulationSettings, clock: FixedClock
) -> None:
    manifest = _build(tmp_path, small_simulation, clock)
    conn = connect(snapshots_root(tmp_path) / manifest.snapshot_id / DB_FILENAME, read_only=True)
    try:
        assert sqlite_repositories(conn).catalog.count() == manifest.counts["products"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM stock")
    finally:
        conn.close()


def test_empty_catalog_fails_without_leaving_files(
    tmp_path: Path, small_simulation: SimulationSettings, clock: FixedClock
) -> None:
    bad = RawCatalogRow("x", "", "", "", "", "", "", "", "")
    source = SourceInfo(kind="synthetic", name="empty", sha256="0")

    with pytest.raises(EmptyCatalogError):
        build_snapshot([bad], source, small_simulation, tmp_path, clock)
    assert not snapshots_root(tmp_path).exists()
