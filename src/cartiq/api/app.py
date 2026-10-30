from fastapi import FastAPI

from cartiq import __version__
from cartiq.api.routes import health
from cartiq.container import Container, build_container


def create_app(container: Container | None = None) -> FastAPI:
    """Build the ASGI app. Tests pass a container; uvicorn calls this as a factory."""
    app = FastAPI(title="CartIQ", version=__version__)
    app.state.container = container if container is not None else build_container()
    app.include_router(health.router)
    return app
