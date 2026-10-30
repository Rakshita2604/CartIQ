"""Command-line entry point. Offline jobs (ingest, build, eval) will be added here as commands."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from cartiq.clock import SystemClock
from cartiq.config import Settings
from cartiq.ingestion.bigbasket import read_bigbasket_csv
from cartiq.ingestion.raw import RawCatalogRow
from cartiq.ingestion.synthetic import SYNTHETIC_CATALOG_VERSION, synthetic_catalog_rows
from cartiq.snapshots import (
    SnapshotVerificationError,
    SourceInfo,
    build_snapshot,
    rows_sha256,
    snapshots_root,
    source_sha256,
    verify_snapshot,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


class CatalogSource(StrEnum):
    SYNTHETIC = "synthetic"
    BIGBASKET = "bigbasket"


@app.command()
def serve(
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev only).")] = False,
) -> None:
    """Run the search API."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(
        "cartiq.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.lower(),
        reload=reload,
    )


@app.command("show-config")
def show_config() -> None:
    """Print the effective configuration after file and environment overrides."""
    typer.echo(Settings().model_dump_json(indent=2))


@app.command("build-snapshot")
def build_snapshot_command(
    source: Annotated[
        CatalogSource, typer.Option(help="Catalog source.")
    ] = CatalogSource.SYNTHETIC,
    input_path: Annotated[
        Path | None,
        typer.Option("--input", help="BigBasket CSV path (defaults to ingestion.bigbasket_csv)."),
    ] = None,
) -> None:
    """Ingest a catalog, simulate stores/inventory/users/orders, and write a snapshot."""
    settings = Settings()
    rows: list[RawCatalogRow]
    if source is CatalogSource.SYNTHETIC:
        rows = synthetic_catalog_rows()
        info = SourceInfo(
            kind="synthetic", name=SYNTHETIC_CATALOG_VERSION, sha256=rows_sha256(rows)
        )
    else:
        path = input_path or settings.ingestion.bigbasket_csv
        if not path.is_file():
            raise typer.BadParameter(
                f"{path} not found. Download 'BigBasket Products.csv' from Kaggle "
                "(surajjha101/bigbasket-entire-product-list-28k-datapoints, CC BY-NC-SA 4.0).",
                param_hint="--input",
            )
        rows = list(read_bigbasket_csv(path))
        info = SourceInfo(kind="bigbasket", name=path.name, sha256=source_sha256(path))

    manifest = build_snapshot(
        rows, info, settings.simulation, settings.paths.artifacts_dir, SystemClock()
    )
    typer.echo(f"snapshot: {manifest.snapshot_id}")
    typer.echo(f"path:     {snapshots_root(settings.paths.artifacts_dir) / manifest.snapshot_id}")
    typer.echo(f"counts:   {manifest.counts}")
    typer.echo(f"ingest:   {manifest.ingestion_report}")


@app.command("verify-snapshot")
def verify_snapshot_command(snapshot_id: str) -> None:
    """Recompute a snapshot's content hash and compare it with its manifest."""
    snapshot_dir = snapshots_root(Settings().paths.artifacts_dir) / snapshot_id
    if not snapshot_dir.is_dir():
        raise typer.BadParameter(f"no snapshot at {snapshot_dir}", param_hint="SNAPSHOT_ID")
    try:
        manifest = verify_snapshot(snapshot_dir)
    except SnapshotVerificationError as error:
        typer.echo(f"FAILED: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"OK: {manifest.snapshot_id} ({manifest.content_sha256})")
