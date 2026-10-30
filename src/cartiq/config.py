"""Application settings loaded from TOML, overridable by CARTIQ_* environment variables."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_FILE_ENV_VAR = "CARTIQ_CONFIG_FILE"
DEFAULT_CONFIG_FILE = Path("config/default.toml")

Environment = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class PathsSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")


class ApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CARTIQ_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    environment: Environment = "dev"
    log_level: LogLevel = "INFO"
    paths: PathsSettings = PathsSettings()
    api: ApiSettings = ApiSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml = TomlConfigSettingsSource(settings_cls, toml_file=resolve_config_file())
        return (init_settings, env_settings, toml)


def resolve_config_file() -> Path | None:
    """Return the TOML file to load, or None to use code defaults only.

    An explicitly configured file must exist; a missing default file is tolerated.
    """
    explicit = os.environ.get(CONFIG_FILE_ENV_VAR)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{CONFIG_FILE_ENV_VAR} points to missing file: {path}")
        return path
    return DEFAULT_CONFIG_FILE if DEFAULT_CONFIG_FILE.is_file() else None
