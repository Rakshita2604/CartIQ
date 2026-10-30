"""Load and validate hand-labeled judgment files (schema version 2).

A judgment file is TOML bound to one catalog by name and fingerprint::

    schema_version = 2
    catalog = "synthetic-v1"
    catalog_sha256 = "..."         # cartiq.catalog_fingerprint of that catalog
    labeler = "manual-v1"

    [[query]]
    id = "amb-oil"
    text = "oil"
    segment = "ambiguous"          # one primary segment
    attributes = []                # extra QueryAttribute values (segment's own is implied)
    coverage = "exhaustive"        # or "partial" (default): unlisted products are unjudged
    exact = ["syn-0046"]           # Grade.EXACT
    relevant = ["syn-0139"]        # Grade.RELEVANT
    related = []                   # Grade.RELATED
    irrelevant = []                # explicitly judged Grade.IRRELEVANT

    [[query.interpretation]]       # for multi-intent queries
    name = "cooking oil"
    share = "dominant"             # dominant | minority | fringe
    products = ["syn-0046"]

    [query.reasons]                # optional: why a product got grade 0, 1 or 2
    "syn-0139" = "minority interpretation: hair oil"

The grade lists are the only source of grades; reasons, interpretations and attributes
explain and constrain them. Structural rules live in ``JudgedQuery``; this module parses,
checks file-level uniqueness, and ties the file to a catalog.
"""

import hashlib
import tomllib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cartiq.domain import (
    Coverage,
    Grade,
    Interpretation,
    InterpretationShare,
    JudgedQuery,
    QueryAttribute,
    QuerySegment,
)

SCHEMA_VERSION = 2


class JudgmentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class JudgmentSet:
    catalog: str
    catalog_sha256: str
    labeler: str
    queries: tuple[JudgedQuery, ...]
    sha256: str


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _InterpretationEntry(_Model):
    name: str
    share: InterpretationShare
    products: list[str] = Field(default_factory=list)


class _QueryEntry(_Model):
    id: str
    text: str
    segment: QuerySegment
    attributes: list[QueryAttribute] = Field(default_factory=list)
    coverage: Coverage = Coverage.PARTIAL
    exact: list[str] = Field(default_factory=list)
    relevant: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    irrelevant: list[str] = Field(default_factory=list)
    interpretation: list[_InterpretationEntry] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)


class _JudgmentFile(_Model):
    schema_version: int
    catalog: str = Field(min_length=1)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labeler: str = Field(min_length=1)
    query: list[_QueryEntry] = Field(min_length=1)


def load_judgments(path: Path) -> JudgmentSet:
    raw = path.read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
        version = document.get("schema_version")
        if version != SCHEMA_VERSION:
            raise JudgmentError(f"{path}: schema_version {version!r}, expected {SCHEMA_VERSION}")
        parsed = _JudgmentFile.model_validate(document)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError) as error:
        raise JudgmentError(f"{path}: {error}") from error

    queries: list[JudgedQuery] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for entry in parsed.query:
        normalized_text = " ".join(entry.text.casefold().split())
        if entry.id in seen_ids:
            raise JudgmentError(f"{path}: duplicate query id {entry.id!r}")
        if normalized_text in seen_texts:
            raise JudgmentError(f"{path}: duplicate query text {entry.text!r}")
        seen_ids.add(entry.id)
        seen_texts.add(normalized_text)
        queries.append(_to_judged_query(entry, parsed.labeler, path))

    return JudgmentSet(
        catalog=parsed.catalog,
        catalog_sha256=parsed.catalog_sha256,
        labeler=parsed.labeler,
        queries=tuple(queries),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def check_against_catalog(
    judgments: JudgmentSet, catalog: str, fingerprint: str, product_ids: Collection[str]
) -> None:
    """Raise unless the judgments were made against exactly this catalog."""
    if judgments.catalog != catalog:
        raise JudgmentError(f"judgments are for catalog {judgments.catalog!r}, not {catalog!r}")
    if judgments.catalog_sha256 != fingerprint:
        raise JudgmentError(
            f"catalog fingerprint mismatch: judgments have {judgments.catalog_sha256}, the "
            f"catalog is {fingerprint}. Re-check the labels against the changed catalog, then "
            "update catalog_sha256."
        )
    unknown = sorted(
        {pid for query in judgments.queries for pid in query.grades if pid not in product_ids}
    )
    if unknown:
        shown = ", ".join(unknown[:10])
        more = f" (+{len(unknown) - 10} more)" if len(unknown) > 10 else ""
        raise JudgmentError(f"judgments reference unknown products: {shown}{more}")


def _to_judged_query(entry: _QueryEntry, labeler: str, path: Path) -> JudgedQuery:
    grades: dict[str, Grade] = {}
    for grade, product_ids in (
        (Grade.EXACT, entry.exact),
        (Grade.RELEVANT, entry.relevant),
        (Grade.RELATED, entry.related),
        (Grade.IRRELEVANT, entry.irrelevant),
    ):
        for product_id in product_ids:
            if product_id in grades:
                raise JudgmentError(f"{path}: {entry.id}: {product_id} is graded more than once")
            grades[product_id] = grade
    try:
        return JudgedQuery(
            query_id=entry.id,
            text=entry.text,
            segment=entry.segment,
            grades=grades,
            labeler=labeler,
            coverage=entry.coverage,
            attributes=frozenset(entry.attributes),
            interpretations=tuple(
                Interpretation(item.name, item.share, frozenset(item.products))
                for item in entry.interpretation
            ),
            reasons=entry.reasons,
        )
    except ValueError as error:
        raise JudgmentError(f"{path}: {entry.id}: {error}") from error
