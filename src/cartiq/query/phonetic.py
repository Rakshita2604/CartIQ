"""Phonetic codes for spelling tie-breaks.

Algorithm: American Soundex (as used by the US census). A code is the word's first letter
followed by three digits for its consonant sounds:

    b f p v -> 1   c g j k q s x z -> 2   d t -> 3   l -> 4   m n -> 5   r -> 6

Vowels (a e i o u y) separate repeated sounds; h and w do not. Adjacent letters with the
same digit count once, and the code is padded with zeros or cut to four characters:
"magii" and "maggi" are both M200, "magic" and "magix" are M220.

Phonetic distance is the edit distance between two codes (0 = same sound pattern).

Limitations: Soundex models English pronunciation, so it is crude for Hindi words and
brand names; it ignores vowels entirely (so it cannot separate "dal" from "dil"); it
keeps the first letter as is (a typo in the first letter changes the code); it looks at
four characters only; letters outside a-z (accents, digits) are ignored. It is therefore
used only to break ties between candidates that are already equally close by edit
distance and brand preference, never to find candidates.
"""

from cartiq.query.spelling_distance import osa_distance

_DIGITS = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}
_SEPARATORS = frozenset("aeiouy")


def soundex(word: str) -> str:
    letters = [ch for ch in word.lower() if "a" <= ch <= "z"]
    if not letters:
        return ""
    code = [letters[0].upper()]
    previous = _DIGITS.get(letters[0], "")
    for letter in letters[1:]:
        digit = _DIGITS.get(letter)
        if digit is None:
            if letter in _SEPARATORS:
                previous = ""  # a vowel lets the next consonant repeat its digit
            continue  # h and w keep the previous digit
        if digit != previous:
            code.append(digit)
        previous = digit
    return ("".join(code) + "000")[:4]


def phonetic_distance(a: str, b: str) -> int:
    return osa_distance(soundex(a), soundex(b))
