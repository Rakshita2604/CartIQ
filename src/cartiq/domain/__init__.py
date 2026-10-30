"""Domain entities and value objects. This package imports nothing else from cartiq."""

from cartiq.domain.catalog import Dimension, Product, Quantity, Unit
from cartiq.domain.evaluation import Grade, Judgment, QuerySegment
from cartiq.domain.inventory import StockRecord, Store
from cartiq.domain.orders import Order, OrderLine, User
from cartiq.domain.search import (
    Candidate,
    Constraint,
    MatchTier,
    ParsedQuery,
    ScoredProduct,
    Substitute,
    SubstituteReason,
)

__all__ = [
    "Candidate",
    "Constraint",
    "Dimension",
    "Grade",
    "Judgment",
    "MatchTier",
    "Order",
    "OrderLine",
    "ParsedQuery",
    "Product",
    "Quantity",
    "QuerySegment",
    "ScoredProduct",
    "StockRecord",
    "Store",
    "Substitute",
    "SubstituteReason",
    "Unit",
    "User",
]
