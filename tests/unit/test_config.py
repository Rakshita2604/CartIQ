from pathlib import Path

import pytest
from pydantic import ValidationError

from cartiq.config import CONFIG_FILE_ENV_VAR, Settings


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cartiq.toml"
    path.write_text('environment = "prod"\n\n[api]\nport = 9100\n')
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(path))
    return path


def test_repo_default_config_loads() -> None:
    settings = Settings()

    assert settings.environment == "dev"
    assert settings.api.port == 8000
    assert settings.paths.artifacts_dir == Path("artifacts")


@pytest.mark.usefixtures("config_file")
def test_values_come_from_configured_toml_file() -> None:
    settings = Settings()

    assert settings.environment == "prod"
    assert settings.api.port == 9100
    assert settings.api.host == "127.0.0.1"  # unspecified keys keep code defaults


@pytest.mark.usefixtures("config_file")
def test_env_vars_override_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTIQ_ENVIRONMENT", "test")
    monkeypatch.setenv("CARTIQ_API__PORT", "9200")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.api.port == 9200


@pytest.mark.usefixtures("config_file")
def test_explicit_arguments_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTIQ_ENVIRONMENT", "test")

    assert Settings(environment="dev").environment == "dev"


def test_missing_explicit_config_file_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(tmp_path / "missing.toml"))

    with pytest.raises(FileNotFoundError, match=r"missing\.toml"):
        Settings()


def test_unknown_keys_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "typo.toml"
    path.write_text("[api]\nprot = 9000\n")
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(path))

    with pytest.raises(ValidationError):
        Settings()


def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARTIQ_API__PORT", "70000")

    with pytest.raises(ValidationError):
        Settings()
