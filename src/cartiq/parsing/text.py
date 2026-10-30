import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Runs of Unicode letters or digits; everything else (punctuation, symbols, "_") separates.
_TOKEN = re.compile(r"[^\W_]+")

# Bump when tokenize() changes behaviour: it is part of every retrieval index identity.
TOKENIZER_VERSION = "v1"


def clean_text(value: str) -> str:
    """NFKC-normalize, collapse whitespace and strip. Case is preserved."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def tokenize(text: str) -> list[str]:
    """Split text into lowercase letter/digit runs, shared by indexing and querying.

    NFKC then casefold, so composed/decomposed accents, full-width forms and case variants
    give the same tokens. Duplicates and order are kept (term frequency needs them). This is
    tokenization only: no stemming, stop words, synonyms or spelling correction.
    """
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())
