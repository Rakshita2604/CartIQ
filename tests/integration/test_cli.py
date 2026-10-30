import json
import re
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


@pytest.mark.usefixtures("artifacts")
def test_evaluate_baseline_on_synthetic_snapshot(artifacts: Path) -> None:
    snapshot_id = _snapshot_id(runner.invoke(app, ["build-snapshot"]).output)

    result = runner.invoke(app, ["evaluate", snapshot_id])
    again = runner.invoke(app, ["evaluate", snapshot_id])

    assert result.exit_code == 0, result.output
    assert "overall" in result.output
    reports = list((artifacts / "eval" / snapshot_id).glob("token-overlap-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["catalog"] == "synthetic-v1"
    assert len(report["stores"]) == 5
    assert report["overall"]["queries"] > 0
    assert again.output == result.output


@pytest.mark.usefixtures("artifacts")
def test_evaluate_rejects_judgments_for_another_catalog() -> None:
    sample = Path(__file__).parents[1] / "fixtures" / "bigbasket_sample.csv"
    built = runner.invoke(app, ["build-snapshot", "--source", "bigbasket", "--input", str(sample)])

    result = runner.invoke(app, ["evaluate", _snapshot_id(built.output)])

    assert result.exit_code == 1
    assert "judgments are for catalog 'synthetic-v1'" in result.output


@pytest.mark.usefixtures("artifacts")
def test_evaluate_rejects_a_changed_catalog(tmp_path: Path) -> None:
    snapshot_id = _snapshot_id(runner.invoke(app, ["build-snapshot"]).output)
    judgments = Path("eval/judgments/synthetic-v1.toml").read_text()
    stale = tmp_path / "stale.toml"
    stale.write_text(
        re.sub(r'catalog_sha256 = "[0-9a-f]{64}"', f'catalog_sha256 = "{"0" * 64}"', judgments)
    )

    result = runner.invoke(app, ["evaluate", snapshot_id, "--judgments", str(stale)])

    assert result.exit_code == 1
    assert "catalog fingerprint mismatch" in result.output


@pytest.mark.usefixtures("artifacts")
def test_snapshot_records_catalog_fingerprint() -> None:
    built = runner.invoke(app, ["build-snapshot"])
    verified = runner.invoke(app, ["verify-snapshot", _snapshot_id(built.output)])

    fingerprint = built.output.splitlines()[2].removeprefix("catalog:").strip()
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
    assert f"catalog fingerprint: {fingerprint}" in verified.output


@pytest.mark.usefixtures("artifacts")
def test_bm25_evaluation_compare_and_benchmark(artifacts: Path) -> None:
    snapshot_id = _snapshot_id(runner.invoke(app, ["build-snapshot"]).output)
    baseline = runner.invoke(app, ["evaluate", snapshot_id])
    bm25 = runner.invoke(app, ["evaluate", snapshot_id, "--ranker", "bm25"])
    reports = sorted((artifacts / "eval" / snapshot_id).glob("*.json"))

    compared = runner.invoke(app, ["compare-reports", str(reports[1]), str(reports[0])])
    bench = runner.invoke(app, ["bench-bm25", snapshot_id, "--repeat", "1"])

    assert baseline.exit_code == bm25.exit_code == 0, bm25.output
    assert [path.name.split("-")[0] for path in reports] == ["bm25f", "token"]
    assert compared.exit_code == 0, compared.output
    assert "before=token-overlap  after=bm25f-" in compared.output
    assert bench.exit_code == 0, bench.output
    assert "index build:" in bench.output
    assert "store search (limit 10): n=" in bench.output


@pytest.mark.usefixtures("artifacts")
def test_compare_reports_refuses_mismatched_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_id = _snapshot_id(runner.invoke(app, ["build-snapshot"]).output)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    runner.invoke(app, ["evaluate", snapshot_id, "--output", str(first)])
    monkeypatch.setenv("CARTIQ_EVALUATION__K", "5")
    runner.invoke(app, ["evaluate", snapshot_id, "--output", str(second)])

    result = runner.invoke(app, ["compare-reports", str(first), str(second)])

    assert result.exit_code == 1
    assert "k: 10 != 5" in result.output
