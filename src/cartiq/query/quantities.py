"""Find the quantity a shopper asked for, and the text left for retrieval without it.

Recognised, in order of precedence:

1. A size, optionally a multi-pack: "500 ml", "1 litre", "6 x 300 ml", "12 pcs" (the
   catalog parser, ``parsing.quantity``), plus a separate "N pack" elsewhere in the query.
   "pack of N" alone gives a pack count without a size.
2. "dozen" / "N dozen" (12 pieces each).
3. "N pack", "N-pack", "N pk", "N packets": a pack count without a size.
4. A trailing bare integer from 2 to ``bare_count_max`` ("eggs 12"). This is recorded as
   an *ambiguous* count (N pieces or a pack of N), never silently as a pack count. 1 and
   larger numbers are left alone ("3 in 1", "colgate 360").

Number words count only directly before a unit, pack or dozen ("do kilo" = 2 kg,
"aadha kilo" = 0.5 kg, "half dozen" = 6 pcs); on their own they are ordinary words ("do"
is also English). Text naming two different sizes is ambiguous and yields no quantity, as
in catalog parsing.

The quantity's spans are removed from the retrieval text, so its numbers and units cannot
match unrelated products ("12" in "Banana 12 pcs"). Spans are offsets into the given text.
"""

import re
from dataclasses import dataclass

from cartiq.domain import Quantity, QuantityKind, QuantityMention, Unit
from cartiq.parsing.quantity import UNIT_WORDS, parse_quantity

NUMBER_WORDS = {
    "ek": "1",
    "one": "1",
    "do": "2",
    "two": "2",
    "teen": "3",
    "three": "3",
    "char": "4",
    "chaar": "4",
    "four": "4",
    "paanch": "5",
    "panch": "5",
    "five": "5",
    "aadha": "0.5",
    "adha": "0.5",
    "aadhi": "0.5",
    "half": "0.5",
    "dedh": "1.5",
    "dhai": "2.5",
    "dhaai": "2.5",
}
_PACK_WORD = r"(?:packets?|packs?|pkts?|pk)"
_COUNTED = sorted(
    {*UNIT_WORDS, "dozen", "dozens", "packets", "packet", "packs", "pack", "pkts", "pkt", "pk"},
    key=len,
    reverse=True,
)
_NUMBER_WORD = re.compile(
    r"(?<![\w])("
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")\s+(?=(?:"
    + "|".join(map(re.escape, _COUNTED))
    + r")\b)"
)
_DOZEN = re.compile(r"(?<![\w.])(?:(\d+(?:\.\d+)?)\s*)?dozens?\b")
_PACK = re.compile(rf"(?<![\w.])(\d+)\s*-?\s*{_PACK_WORD}\b")
_BARE = re.compile(r"(?<![\w.])(\d+)\s*$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class QuantityExtraction:
    mention: QuantityMention | None
    retrieval_text: str


def extract_quantity(text: str, *, bare_count_max: int) -> QuantityExtraction:
    lowered, starts, ends = _substitute_number_words(text)

    def to_text(span: tuple[int, int]) -> tuple[int, int]:
        return starts[span[0]], ends[span[1] - 1]

    mention = _find(lowered, bare_count_max)
    if mention is None:
        return QuantityExtraction(None, _collapse(text))
    kind, spans, quantity, pack_count, count = mention
    original = tuple(sorted(to_text(span) for span in spans))
    found = QuantityMention(
        kind=kind,
        spans=original,
        text=" ".join(text[start:end] for start, end in original),
        quantity=quantity,
        pack_count=pack_count,
        count=count,
    )
    remaining = list(text)
    for start, end in original:
        remaining[start:end] = " " * (end - start)
    return QuantityExtraction(found, _collapse("".join(remaining)))


_Found = tuple[QuantityKind, list[tuple[int, int]], Quantity | None, int | None, int | None]


def _find(lowered: str, bare_count_max: int) -> _Found | None:
    amount = parse_quantity(lowered)
    if amount is not None and amount.quantity is not None:
        spans = [(amount.start, amount.end)]
        pack_count = amount.pack_count if amount.pack_count > 1 else None
        if pack_count is None:
            pack = next(
                (m for m in _PACK.finditer(lowered) if not _overlaps(m.span(), spans[0])), None
            )
            if pack is not None and int(pack[1]) > 1:
                pack_count = int(pack[1])
                spans.append(pack.span())
        return QuantityKind.AMOUNT, spans, amount.quantity, pack_count, None
    if amount is not None:  # "pack of N" without a size
        return QuantityKind.PACK, [(amount.start, amount.end)], None, amount.pack_count, None

    dozen = _DOZEN.search(lowered)
    if dozen is not None:
        pieces = float(dozen[1] or 1) * 12
        if pieces.is_integer() and pieces > 0:
            quantity = Quantity(pieces, Unit.PCS)
            return QuantityKind.AMOUNT, [dozen.span()], quantity, None, None
    pack = _PACK.search(lowered)
    if pack is not None and int(pack[1]) > 1:
        return QuantityKind.PACK, [pack.span()], None, int(pack[1]), None
    bare = _BARE.search(lowered)
    if bare is not None and 2 <= int(bare[1]) <= bare_count_max and lowered[: bare.start()].strip():
        return QuantityKind.BARE_NUMBER, [bare.span(1)], None, None, int(bare[1])
    return None


def _substitute_number_words(text: str) -> tuple[str, list[int], list[int]]:
    """Lowercase ``text`` and replace number words before a unit with digits.

    Returns the new text and, for each of its characters, the start and end offset of the
    original text it came from, so spans can be mapped back.
    """
    lowered = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    position = 0
    for match in _NUMBER_WORD.finditer(lowered):
        word_start, word_end = match.span(1)
        for i in range(position, word_start):
            chars.append(lowered[i])
            starts.append(i)
            ends.append(i + 1)
        for digit in NUMBER_WORDS[match[1]]:
            chars.append(digit)
            starts.append(word_start)
            ends.append(word_end)
        position = word_end
    for i in range(position, len(lowered)):
        chars.append(lowered[i])
        starts.append(i)
        ends.append(i + 1)
    return "".join(chars), starts, ends


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
