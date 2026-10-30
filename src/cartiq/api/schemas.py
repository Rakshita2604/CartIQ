"""HTTP response schemas. Kept separate from domain entities so the wire format can evolve."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cartiq.config import Environment


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    version: str
    environment: Environment
    started_at: datetime
    uptime_seconds: float
