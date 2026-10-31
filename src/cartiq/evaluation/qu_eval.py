"""Score query understanding against the hand-written expectations file.

Measured per suite (one catalog): quantity exact-match accuracy, brand precision and recall
(by brand key), expected-correction coverage, unexpected corrections (listed, including in
cases whose expected correction is covered), do-not-correct violations and forbidden terms.
Expected values never come from parser output; the report holds no timestamps.
"""

import hashlib
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cartiq.domain import ParsedQuery, Quantity, QuantityKind, TermSource, Unit
from cartiq.query.brands import brand_key


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QuantityExpectation(_Strict):
    kind: Literal["none", "amount", "pack", "bare_number"]
    value: float | None = None
    unit: Unit | None = None
    pack_count: int | None = None
    count: int | None = None


class CorrectionExpectation(_Strict):
    typed: str
    intended: str


class CaseExpectation(_Strict):
    text: str
    quantity: QuantityExpectation
    brand: str
    corrections: list[CorrectionExpectation]


class SuiteExpectation(_Strict):
    name: str
    catalog: str
    catalog_sha256: str
    do_not_correct: list[str]
    never_produce: list[str]
    case: list[CaseExpectation]


class ExpectationFile(_Strict):
    schema_version: Literal[1]
    version: str
    labeler: str
    suite: list[SuiteExpectation]


def load_expectations(path: Path) -> tuple[ExpectationFile, str]:
    raw = path.read_bytes()
    return ExpectationFile.model_validate(tomllib.loads(raw.decode("utf-8"))), hashlib.sha256(
        raw
    ).hexdigest()


BrandOutcome = Literal["tp", "fp", "fn", "tn", "wrong"]


class CoverageResult(_Strict):
    typed: str
    intended: str
    covered: bool


class CaseResult(_Strict):
    text: str
    quantity_ok: bool
    expected_quantity: str
    actual_quantity: str
    expected_brand: str
    actual_brand: str
    brand_outcome: BrandOutcome
    coverage: list[CoverageResult]
    unexpected_corrections: list[str]
    brand_notes: list[str]


class SuiteReport(_Strict):
    suite: str
    catalog: str
    pipeline: str
    expectations_sha256: str
    cases: int
    quantity_correct: int
    brand_tp: int
    brand_fp: int
    brand_fn: int
    brand_precision: float | None
    brand_recall: float | None
    corrections_expected: int
    corrections_covered: int
    unexpected_corrections: list[str]  # "query: typed->target (reason)"
    do_not_correct_violations: list[str]
    never_produce_violations: list[str]
    case_results: list[CaseResult]


def evaluate_suite(
    suite: SuiteExpectation,
    parse: Callable[[str], ParsedQuery],
    workload: Sequence[str],
    *,
    pipeline: str,
    expectations_sha256: str,
) -> SuiteReport:
    """``workload`` is extra query text (the development queries) checked for
    ``never_produce`` terms."""
    results = [_evaluate_case(case, parse(case.text)) for case in suite.case]
    tp = sum(r.brand_outcome == "tp" for r in results)
    fp = sum(r.brand_outcome in ("fp", "wrong") for r in results)
    fn = sum(r.brand_outcome in ("fn", "wrong") for r in results)
    forbidden = set(suite.never_produce)
    violations = sorted(
        {
            f"{text!r} produced {term!r}"
            for text in [*(c.text for c in suite.case), *workload]
            for concept in parse(text).concepts
            for term in (t.term for t in concept.terms)
            if term in forbidden
        }
    )
    return SuiteReport(
        suite=suite.name,
        catalog=suite.catalog,
        pipeline=pipeline,
        expectations_sha256=expectations_sha256,
        cases=len(results),
        quantity_correct=sum(r.quantity_ok for r in results),
        brand_tp=tp,
        brand_fp=fp,
        brand_fn=fn,
        brand_precision=tp / (tp + fp) if tp + fp else None,
        brand_recall=tp / (tp + fn) if tp + fn else None,
        corrections_expected=sum(len(r.coverage) for r in results),
        corrections_covered=sum(c.covered for r in results for c in r.coverage),
        unexpected_corrections=[
            f"{r.text!r}: {item}" for r in results for item in r.unexpected_corrections
        ],
        do_not_correct_violations=[
            f"{word}->{fix.corrected} ({fix.reason.value})"
            for word in suite.do_not_correct
            for fix in parse(word).corrections
        ],
        never_produce_violations=violations,
        case_results=results,
    )


def format_suite_report(report: SuiteReport) -> str:
    def ratio(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    lines = [
        f"suite={report.suite} catalog={report.catalog} pipeline={report.pipeline} "
        f"expectations={report.expectations_sha256[:12]}",
        f"quantity:  {report.quantity_correct}/{report.cases} exact",
        f"brand:     precision {ratio(report.brand_precision)}  recall {ratio(report.brand_recall)}"
        f"  (tp {report.brand_tp}, fp {report.brand_fp}, fn {report.brand_fn})",
        f"spelling:  expected corrections covered {report.corrections_covered}/"
        f"{report.corrections_expected}; unexpected corrections "
        f"{len(report.unexpected_corrections)}",
        *(f"           unexpected {item}" for item in report.unexpected_corrections),
        f"do-not-correct violations: {report.do_not_correct_violations or 'none'}",
        f"never-produce violations:  {report.never_produce_violations or 'none'}",
    ]
    for r in report.case_results:
        problems = []
        if not r.quantity_ok:
            problems.append(f"quantity {r.actual_quantity} (expected {r.expected_quantity})")
        if r.brand_outcome in ("fp", "fn", "wrong"):
            problems.append(
                f"brand {r.brand_outcome}: {r.actual_brand or '-'} "
                f"(expected {r.expected_brand or '-'})"
            )
        problems += [f"missed {c.typed}->{c.intended}" for c in r.coverage if not c.covered]
        if problems or r.brand_notes:
            lines.append(f"  {r.text!r}: {'; '.join(problems) or 'ok'}")
            lines += [f"      note: {note}" for note in r.brand_notes]
    return "\n".join(lines)


def _evaluate_case(case: CaseExpectation, parsed: ParsedQuery) -> CaseResult:
    expected_key = brand_key(case.brand) if case.brand else ""
    actual_key = parsed.brand_key or ""
    outcome: BrandOutcome
    if not expected_key and not actual_key:
        outcome = "tn"
    elif expected_key == actual_key:
        outcome = "tp"
    elif not expected_key:
        outcome = "fp"
    elif not actual_key:
        outcome = "fn"
    else:
        outcome = "wrong"

    coverage = []
    for fix in case.corrections:
        concept = next(
            (
                c
                for c in parsed.concepts
                if any(t.term == fix.typed and t.source is TermSource.QUERY for t in c.terms)
            ),
            None,
        )
        terms = {t.term for t in concept.terms} if concept else set()
        coverage.append(
            CoverageResult(typed=fix.typed, intended=fix.intended, covered=fix.intended in terms)
        )
    expected_pairs = {(fix.typed, fix.intended) for fix in case.corrections}
    unexpected = [
        f"{fix.original}->{fix.corrected} ({fix.reason.value})"
        for fix in parsed.corrections
        if (fix.original, fix.corrected) not in expected_pairs
    ]
    return CaseResult(
        text=case.text,
        quantity_ok=_quantity_matches(case.quantity, parsed),
        expected_quantity=_describe_expected(case.quantity),
        actual_quantity=_describe_actual(parsed),
        expected_brand=case.brand,
        actual_brand=parsed.brand or "",
        brand_outcome=outcome,
        coverage=coverage,
        unexpected_corrections=unexpected,
        brand_notes=list(parsed.brand_notes),
    )


def _quantity_matches(expected: QuantityExpectation, parsed: ParsedQuery) -> bool:
    mention = parsed.quantity_mention
    if expected.kind == "none":
        return mention is None
    if mention is None or mention.kind is not QuantityKind(expected.kind):
        return False
    if expected.kind == "amount":
        if expected.value is None or expected.unit is None:
            return False
        return (
            mention.quantity is not None
            and mention.quantity.same_amount(Quantity(expected.value, expected.unit))
            and mention.pack_count == expected.pack_count
        )
    if expected.kind == "pack":
        return mention.pack_count == expected.pack_count
    return mention.count == expected.count


def _describe_expected(expected: QuantityExpectation) -> str:
    fields = expected.model_dump(exclude_none=True, mode="json")
    return ",".join(f"{k}={v}" for k, v in fields.items())


def _describe_actual(parsed: ParsedQuery) -> str:
    mention = parsed.quantity_mention
    if mention is None:
        return "kind=none"
    parts = [f"kind={mention.kind.value}"]
    if mention.quantity is not None:
        parts += [f"value={mention.quantity.value:g}", f"unit={mention.quantity.unit.value}"]
    if mention.pack_count is not None:
        parts.append(f"pack_count={mention.pack_count}")
    if mention.count is not None:
        parts.append(f"count={mention.count}")
    return ",".join(parts)
