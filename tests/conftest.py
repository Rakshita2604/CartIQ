import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from cartiq.api.app import create_app
from cartiq.clock import FixedClock
from cartiq.config import Settings
from cartiq.container import build_container

T0 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's CARTIQ_* environment from leaking into tests."""
    for key in list(os.environ):
        if key.startswith("CARTIQ_"):
            monkeypatch.delenv(key)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(T0)


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="test")


@pytest.fixture
def client(settings: Settings, clock: FixedClock) -> Iterator[TestClient]:
    with TestClient(create_app(build_container(settings, clock))) as test_client:
        yield test_client
