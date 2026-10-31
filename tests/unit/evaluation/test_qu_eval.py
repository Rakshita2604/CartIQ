from collections.abc import Sequence
from pathlib import Path

from cartiq.domain import (
    ConceptTerm,
    Correction,
    CorrectionReason,
    ParsedQuery,
    Quantity,
    QuantityKind,
    QuantityMention,
    QueryConcept,
    TermSource,
    Unit,
)
from cartiq.evaluation import load_judgments
from cartiq.evaluation.qu_eval import (
    CaseExpectation,
    CorrectionExpectation,
    QuantityExpectation,
    SuiteExpectation,
    SuiteReport,
    evaluate_suite,
    format_suite_report,
    load_expectations,
)

NONE = QuantityExpectation(kind="none")


def case(
    text: str,
    brand: str = "",
    quantity: QuantityExpectation = NONE,
    fixes: Sequence[tuple[str, str]] = (),
) -> CaseExpectation:
    return CaseExpectation(
        text=text,
        quantity=quantity,
        brand=brand,
        corrections=[CorrectionExpectation(typed=t, intended=i) for t, i in fixes],
    )


def parsed(
    text: str,
    *,
    brand: str | None = None,
    concepts: Sequence[QueryConcept] = (),
    corrections: Sequence[Correction] = (),
    mention: QuantityMention | None = None,
) -> ParsedQuery:
    return ParsedQuery(
        raw=text,
        normalized=text,
        tokens=tuple(text.split()),
        concepts=tuple(concepts),
        corrections=tuple(corrections),
        quantity_mention=mention,
        brand=brand,
        brand_key=None if brand is None else brand.casefold().replace("-", ""),
    )


BISKUT = QueryConcept(
    "biscut",
    (
        ConceptTerm("biscut", TermSource.QUERY),
        ConceptTerm("biskut", TermSource.SPELLING),
        ConceptTerm("biscuit", TermSource.LEXICON),
    ),
)
OUTPUTS = {
    "biscut": parsed(
        "biscut",
        concepts=[BISKUT],
        corrections=[Correction("biscut", "biskut", 1, CorrectionReason.UNKNOWN)],
    ),
    "coke 300 ml": parsed(
        "coke 300 ml",
        brand="Coca-Cola",
        concepts=[QueryConcept.single("coke"), QueryConcept.single("come")],
        mention=QuantityMention(QuantityKind.AMOUNT, ((5, 11),), "300 ml", Quantity(300, Unit.ML)),
    ),
    "real juice": parsed("real juice", brand="Real"),
    "amul": parsed("amul"),
    "chicken": parsed(
        "chicken", corrections=[Correction("chicken", "chicken2", 1, CorrectionReason.DOMINANT)]
    ),
}
SUITE = SuiteExpectation(
    name="toy",
    catalog="toy",
    catalog_sha256="0" * 64,
    do_not_correct=["chicken", "amul"],
    never_produce=["come"],
    case=[
        case("biscut", fixes=[("biscut", "biscuit")]),
        case(
            "coke 300 ml",
            brand="Coca-Cola",
            quantity=QuantityExpectation(kind="amount", value=0.3, unit=Unit.L),
        ),
        case("real juice"),
        case("amul", brand="Amul"),
    ],
)


def run() -> SuiteReport:
    return evaluate_suite(
        SUITE, lambda text: OUTPUTS[text], ["amul"], pipeline="toy", expectations_sha256="1" * 64
    )


def test_coverage_and_unexpected_corrections_are_separate() -> None:
    report = run()

    # The intended word is reached (through the lexicon) ...
    assert (report.corrections_expected, report.corrections_covered) == (1, 1)
    # ... but the different correction target is still listed as unexpected.
    assert report.unexpected_corrections == ["'biscut': biscut->biskut (unknown)"]


def test_brand_precision_and_recall() -> None:
    report = run()

    assert (report.brand_tp, report.brand_fp, report.brand_fn) == (1, 1, 1)
    assert report.brand_precision == 0.5
    assert report.brand_recall == 0.5
    outcomes = {r.text: r.brand_outcome for r in report.case_results}
    assert outcomes == {"biscut": "tn", "coke 300 ml": "tp", "real juice": "fp", "amul": "fn"}


def test_quantity_equivalent_units_match() -> None:
    report = run()

    assert report.quantity_correct == 4


def test_violations_are_listed() -> None:
    report = run()
    text = format_suite_report(report)

    assert report.do_not_correct_violations == ["chicken->chicken2 (dominant)"]
    assert report.never_produce_violations == ["'coke 300 ml' produced 'come'"]
    assert "brand fp: Real (expected -)" in text


def test_committed_expectations_cover_the_development_queries() -> None:
    document, _ = load_expectations(Path("eval/query_understanding/synthetic-v1.toml"))
    dev = [q.text for q in load_judgments(Path("eval/judgments/synthetic-v1.toml")).queries]
    held = {
        q.text for q in load_judgments(Path("eval/judgments/synthetic-v1-heldout.toml")).queries
    }

    synthetic, bigbasket = document.suite
    assert [c.text for c in synthetic.case] == dev
    assert not ({c.text for s in document.suite for c in s.case} & held)
    assert {c.text for c in bigbasket.case} <= set(dev)
