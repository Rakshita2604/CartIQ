"""Command-line entry point. Offline jobs (ingest, build, eval) will be added here as commands."""

import logging
from typing import Annotated

import typer
import uvicorn

from cartiq.config import Settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


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
