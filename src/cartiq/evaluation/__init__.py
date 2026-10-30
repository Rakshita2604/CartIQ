"""Offline evaluation: judgment sets, ranking metrics and the evaluation runner."""

from cartiq.evaluation.judgments import (
    JudgmentError,
    JudgmentSet,
    check_against_catalog,
    load_judgments,
)
from cartiq.evaluation.metrics import judged_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from cartiq.evaluation.runner import (
    EvalReport,
    SearchPipeline,
    StagedResults,
    evaluate,
    format_report,
)

__all__ = [
    "EvalReport",
    "JudgmentError",
    "JudgmentSet",
    "SearchPipeline",
    "StagedResults",
    "check_against_catalog",
    "evaluate",
    "format_report",
    "judged_at_k",
    "load_judgments",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
