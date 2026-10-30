"""The token-overlap baseline report must stay byte-for-byte identical.

It is the fixed reference every later phase is compared against. The pinned hash changes
only when the snapshot, the judgments or the report format change on purpose; update it
in the same commit and say why.
"""

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cartiq.cli import app

BASELINE_REPORT_SHA256 = "07bcc90b73804102d8a06f6a1341678b15ac588f8d69fe971157ccbad4479197"
SNAPSHOT_ID = "snap-d0141915a979"  # default simulation settings over synthetic-v1


def test_baseline_report_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTIQ_PATHS__ARTIFACTS_DIR", str(tmp_path))
    runner = CliRunner()

    built = runner.invoke(app, ["build-snapshot"])
    report = tmp_path / "baseline.json"
    evaluated = runner.invoke(app, ["evaluate", SNAPSHOT_ID, "--output", str(report)])

    assert built.output.startswith(f"snapshot: {SNAPSHOT_ID}"), built.output
    assert evaluated.exit_code == 0, evaluated.output
    assert hashlib.sha256(report.read_bytes()).hexdigest() == BASELINE_REPORT_SHA256
