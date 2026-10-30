"""Small invariant checks shared by domain entities. Each raises ValueError."""

import math
from datetime import datetime


def require_id(name: str, value: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")


def require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
