import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cartiq.cli import app

runner = CliRunner()


def test_show_config_prints_effective_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTIQ_API__PORT", "9300")

    result = runner.invoke(app, ["show-config"])

    assert result.exit_code == 0, result.output
    config = json.loads(result.output)
    assert config["api"]["port"] == 9300
    assert config["environment"] == "dev"


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CARTIQ_PATHS__ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("CARTIQ_SIMULATION__N_USERS", "20")
    return tmp_path


def _snapshot_id(output: str) -> str:
    return output.splitlines()[0].removeprefix("snapshot:").strip()


@pytest.mark.usefixtures("artifacts")
def test_build_then_verify_synthetic_snapshot() -> None:
    built = runner.invoke(app, ["build-snapshot"])
    assert built.exit_code == 0, built.output
    snapshot_id = _snapshot_id(built.output)

    rebuilt = runner.invoke(app, ["build-snapshot", "--source", "synthetic"])
    verified = runner.invoke(app, ["verify-snapshot", snapshot_id])

    assert _snapshot_id(rebuilt.output) == snapshot_id
    assert verified.exit_code == 0, verified.output
    assert verified.output.startswith(f"OK: {snapshot_id}")


@pytest.mark.usefixtures("artifacts")
def test_build_from_bigbasket_csv() -> None:
    sample = Path(__file__).parents[1] / "fixtures" / "bigbasket_sample.csv"

    result = runner.invoke(app, ["build-snapshot", "--source", "bigbasket", "--input", str(sample)])

    assert result.exit_code == 0, result.output
    assert "'products': 4" in result.output


@pytest.mark.usefixtures("artifacts")
def test_bigbasket_missing_file_explains_where_to_get_it(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"

    result = runner.invoke(
        app, ["build-snapshot", "--source", "bigbasket", "--input", str(missing)]
    )

    assert result.exit_code != 0
    assert "Kaggle" in result.output


@pytest.mark.usefixtures("artifacts")
def test_verify_unknown_snapshot_fails() -> None:
    result = runner.invoke(app, ["verify-snapshot", "snap-000000000000"])

    assert result.exit_code != 0
