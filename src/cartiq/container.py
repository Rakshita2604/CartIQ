"""Composition root: the one place where concrete dependencies are built and wired."""

from dataclasses import dataclass
from datetime import datetime

from cartiq.clock import Clock, SystemClock
from cartiq.config import Settings


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    clock: Clock
    started_at: datetime


def build_container(settings: Settings | None = None, clock: Clock | None = None) -> Container:
    settings = settings if settings is not None else Settings()
    clock = clock if clock is not None else SystemClock()
    return Container(settings=settings, clock=clock, started_at=clock.now())
