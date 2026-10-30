from datetime import timedelta

from fastapi.testclient import TestClient

from cartiq import __version__
from cartiq.clock import FixedClock


def test_health_reports_status_and_metadata(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "environment": "test",
        "started_at": "2026-01-01T09:00:00Z",
        "uptime_seconds": 0.0,
    }


def test_health_uptime_uses_injected_clock(client: TestClient, clock: FixedClock) -> None:
    clock.advance(timedelta(seconds=90))

    assert client.get("/health").json()["uptime_seconds"] == 90.0


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
