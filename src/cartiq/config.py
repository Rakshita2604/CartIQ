"""Application settings loaded from TOML, overridable by CARTIQ_* environment variables."""

import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
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


class IngestionSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bigbasket_csv: Path = Path("data/raw/bigbasket_products.csv")


class SimulationSettings(BaseModel):
    """Parameters of the seeded store/inventory/user/order simulation.

    Every value that affects generated data lives here (no wall-clock reads), so a snapshot is
    fully determined by the catalog plus these settings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42
    n_stores: int = Field(default=5, ge=1)
    cities: tuple[str, ...] = Field(default=("Bengaluru", "Mumbai", "Delhi"), min_length=1)
    assortment_ratio: float = Field(default=0.7, gt=0, le=1)
    out_of_stock_rate: float = Field(default=0.08, ge=0, lt=1)
    max_stock_units: int = Field(default=50, ge=1)
    n_users: int = Field(default=500, ge=0)
    min_orders_per_user: int = Field(default=1, ge=0)
    max_orders_per_user: int = Field(default=12, ge=0)
    max_lines_per_order: int = Field(default=8, ge=1)
    brand_loyalty: float = Field(default=0.75, ge=0, le=1)
    popularity_exponent: float = Field(default=1.0, ge=0)
    history_start: date = date(2026, 1, 1)
    history_days: int = Field(default=90, ge=1)
    inventory_as_of: AwareDatetime = datetime(2026, 4, 1, tzinfo=UTC)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if self.min_orders_per_user > self.max_orders_per_user:
            raise ValueError("min_orders_per_user must be <= max_orders_per_user")
        return self


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
    ingestion: IngestionSettings = IngestionSettings()
    simulation: SimulationSettings = SimulationSettings()

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
