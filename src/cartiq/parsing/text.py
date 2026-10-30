import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    """NFKC-normalize, collapse whitespace and strip. Case is preserved."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
