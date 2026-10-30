from pathlib import Path

import pytest

from cartiq.domain import Coverage, Grade, InterpretationShare, QueryAttribute, QuerySegment
from cartiq.evaluation import JudgmentError, check_against_catalog, load_judgments

FINGERPRINT = "a" * 64

VALID = f"""
schema_version = 2
catalog = "test-catalog"
catalog_sha256 = "{FINGERPRINT}"
labeler = "manual-test"

[[query]]
id = "q-milk"
text = "amul milk"
segment = "brand"
attributes = ["misspelling"]
coverage = "exhaustive"
exact = ["P1", "P2"]
relevant = ["P3"]
related = ["P4"]
irrelevant = ["P5"]

[query.reasons]
"P3" = "other brand"
"P5" = "same brand, butter"

[[query]]
id = "q-oil"
text = "oil"
segment = "ambiguous"
exact = ["P6"]
relevant = ["P7"]

[[query.interpretation]]
name = "cooking oil"
share = "dominant"
products = ["P6"]

[[query.interpretation]]
name = "hair oil"
share = "minority"
products = ["P7"]
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "judgments.toml"
    path.write_text(text)
    return path


def test_loads_every_field(tmp_path: Path) -> None:
    judgments = load_judgments(_write(tmp_path, VALID))

    assert (judgments.catalog, judgments.catalog_sha256) == ("test-catalog", FINGERPRINT)
    milk, oil = judgments.queries
    assert milk.coverage is Coverage.EXHAUSTIVE
    assert milk.attributes == {QueryAttribute.BRAND, QueryAttribute.MISSPELLING}
    assert dict(milk.grades) == {
        "P1": Grade.EXACT,
        "P2": Grade.EXACT,
        "P3": Grade.RELEVANT,
        "P4": Grade.RELATED,
        "P5": Grade.IRRELEVANT,
    }
    assert milk.reasons["P5"] == "same brand, butter"
    assert oil.coverage is Coverage.PARTIAL  # the default
    assert oil.segment is QuerySegment.AMBIGUOUS
    assert [i.share for i in oil.interpretations] == [
        InterpretationShare.DOMINANT,
        InterpretationShare.MINORITY,
    ]
    assert len(judgments.sha256) == 64


def test_hash_changes_with_content(tmp_path: Path) -> None:
    first = load_judgments(_write(tmp_path, VALID)).sha256
    second = load_judgments(_write(tmp_path, VALID.replace('"P4"', '"P9"'))).sha256

    assert first != second


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version = 2", "schema_version = 1", "expected 2"),
        (f'catalog_sha256 = "{FINGERPRINT}"', 'catalog_sha256 = "abc"', "catalog_sha256"),
        ('id = "q-oil"', 'id = "q-milk"', "duplicate query id"),
        ('text = "oil"', 'text = "  AMUL milk "', "duplicate query text"),
        ('related = ["P4"]', 'related = ["P1"]', "graded more than once"),
        ('irrelevant = ["P5"]', 'irrelevant = ["P4"]', "graded more than once"),
        ('segment = "brand"', 'segment = "slang"', "segment"),
        ('attributes = ["misspelling"]', 'attributes = ["typo"]', "attributes"),
        ('coverage = "exhaustive"', 'coverage = "complete"', "coverage"),
        ('share = "minority"', 'share = "rare"', "share"),
        ('"P3" = "other brand"', '"P1" = "exact"', "EXACT"),
        ('products = ["P7"]', "products = []", "P7 belongs to no interpretation"),
        ('labeler = "manual-test"', 'labeler = "manual-test"\nnotes = "x"', "notes"),
        ('catalog = "test-catalog"', 'catalog = "test-catalog', "judgments.toml"),
    ],
)
def test_rejects_invalid_files(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in VALID

    with pytest.raises(JudgmentError, match=message):
        load_judgments(_write(tmp_path, VALID.replace(old, new, 1)))


class TestCheckAgainstCatalog:
    IDS = frozenset(f"P{i}" for i in range(1, 8))

    def test_matching_catalog_passes(self, tmp_path: Path) -> None:
        check_against_catalog(
            load_judgments(_write(tmp_path, VALID)), "test-catalog", FINGERPRINT, self.IDS
        )

    def test_other_catalog_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(JudgmentError, match="not 'other'"):
            check_against_catalog(
                load_judgments(_write(tmp_path, VALID)), "other", FINGERPRINT, self.IDS
            )

    def test_fingerprint_mismatch_is_rejected_and_shows_actual(self, tmp_path: Path) -> None:
        with pytest.raises(JudgmentError, match="the catalog is " + "b" * 64):
            check_against_catalog(
                load_judgments(_write(tmp_path, VALID)), "test-catalog", "b" * 64, self.IDS
            )

    def test_unknown_ids_are_reported(self, tmp_path: Path) -> None:
        with pytest.raises(JudgmentError, match="P6, P7"):
            check_against_catalog(
                load_judgments(_write(tmp_path, VALID)),
                "test-catalog",
                FINGERPRINT,
                self.IDS - {"P6", "P7"},
            )
