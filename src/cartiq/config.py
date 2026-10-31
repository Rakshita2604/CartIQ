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
SpellingField = Literal["name", "product_type", "brand", "sub_category", "description"]
ProductField = Literal["name", "product_type", "brand", "sub_category", "description", "category"]


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


class FieldWeight(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    weight: float = Field(ge=0)  # 0 removes the field from the index entirely
    b: float = Field(ge=0, le=1)  # length normalization strength


class BM25Settings(BaseModel):
    """BM25F parameters. Fixed a priori; not tuned against the evaluation judgments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k1: float = Field(default=1.2, gt=0)
    name: FieldWeight = FieldWeight(weight=3.0, b=0.75)
    product_type: FieldWeight = FieldWeight(weight=2.0, b=0.0)
    brand: FieldWeight = FieldWeight(weight=1.0, b=0.0)
    sub_category: FieldWeight = FieldWeight(weight=0.5, b=0.0)
    description: FieldWeight = FieldWeight(weight=0.3, b=0.75)
    category: FieldWeight = FieldWeight(weight=0.0, b=0.0)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bm25: BM25Settings = BM25Settings()


class QuerySettings(BaseModel):
    """Query understanding. Values are fixed a priori (Phase 4 design), never tuned on
    evaluation judgments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lexicon: str = "hinglish-v1"
    brand_aliases: str = "brand-aliases-v1"
    # Spelling: tokens shorter than one_edit_min_length are never corrected; one edit is
    # allowed from that length, two edits from two_edit_min_length.
    one_edit_min_length: int = Field(default=4, ge=1)
    two_edit_min_length: int = Field(default=8, ge=1)
    # A known token gets a one-edit neighbour added when the neighbour is this dominant.
    dominance_ratio: float = Field(default=20.0, gt=1)
    dominance_min_df: int = Field(default=5, ge=1)
    # A trailing bare integer in [2, bare_count_max] is read as an (ambiguous) count.
    bare_count_max: int = Field(default=50, ge=2)
    # Product fields whose words are spelling targets (Phase 4b: descriptions and
    # sub-categories excluded, so words that only occur in marketing text are never targets).
    spelling_fields: tuple[SpellingField, ...] = Field(
        default=("name", "brand", "product_type"), min_length=1
    )
    # A one-word brand is recognised only if at least this share of the products containing
    # the word (in name, brand, product type) carry that brand ("real" vs brand "Real").
    brand_min_share: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        if self.two_edit_min_length <= self.one_edit_min_length:
            raise ValueError("two_edit_min_length must exceed one_edit_min_length")
        return self


class RankingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Constraint-first ordering (stage s6) reorders this many store-eligible BM25F
    # candidates before cutting to the result limit.
    candidate_depth: int = Field(default=100, ge=1)
    # Product fields that decide whether a product contains a query concept (exactness).
    # Shelf text and descriptions still feed BM25F retrieval, but not exact matching.
    coverage_fields: tuple[ProductField, ...] = Field(
        default=("name", "brand", "product_type"), min_length=1
    )


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    judgments_file: Path = Path("eval/judgments/synthetic-v1.toml")
    k: int = Field(default=10, ge=1)
    recall_at: tuple[int, ...] = Field(default=(10, 100), min_length=1)
    # Lowest grade that counts as a hit for MRR and Recall (2 = relevant, 3 = exact only).
    hit_min_grade: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def _check_depths(self) -> Self:
        if min(self.recall_at) < 1:
            raise ValueError("recall_at depths must be >= 1")
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
    retrieval: RetrievalSettings = RetrievalSettings()
    query: QuerySettings = QuerySettings()
    ranking: RankingSettings = RankingSettings()
    evaluation: EvaluationSettings = EvaluationSettings()

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
