import json

import pytest
from typer.testing import CliRunner

from cartiq.cli import app


def test_show_config_prints_effective_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTIQ_API__PORT", "9300")

    result = CliRunner().invoke(app, ["show-config"])

    assert result.exit_code == 0, result.output
    config = json.loads(result.output)
    assert config["api"]["port"] == 9300
    assert config["environment"] == "dev"
