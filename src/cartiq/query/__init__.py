"""Query understanding: quantity extraction, Hinglish lexicon expansion, spelling
correction, plural variants. Rule-based and deterministic; no models or LLM calls."""

from cartiq.query.parser import STAGES, QueryParser, Stages

__all__ = ["STAGES", "QueryParser", "Stages"]
