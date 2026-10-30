from fastapi import APIRouter

from cartiq import __version__
from cartiq.api.dependencies import ContainerDep
from cartiq.api.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health")
def health(container: ContainerDep) -> HealthResponse:
    uptime = container.clock.now() - container.started_at
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=container.settings.environment,
        started_at=container.started_at,
        uptime_seconds=uptime.total_seconds(),
    )
