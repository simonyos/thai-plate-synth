"""Thai license-plate alphabet.

Class indices are stable across the project: 0..43 are the 44 Thai consonants
in canonical dictionary order, 44..53 are the Arabic digits 0..9. This layout
is emitted by the renderer and consumed by training/eval without remapping.
"""

from __future__ import annotations

CONSONANTS: tuple[str, ...] = (
    "ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ",
    "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ",
    "ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป",
    "ผ", "ฝ", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ล",
    "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ",
)

DIGITS: tuple[str, ...] = tuple("0123456789")

ALPHABET: tuple[str, ...] = CONSONANTS + DIGITS

CLASS_TO_GLYPH: dict[int, str] = {i: g for i, g in enumerate(ALPHABET)}
GLYPH_TO_CLASS: dict[str, int] = {g: i for i, g in enumerate(ALPHABET)}

N_CONSONANTS = len(CONSONANTS)
N_DIGITS = len(DIGITS)
N_CLASSES = len(ALPHABET)

assert N_CONSONANTS == 44
assert N_DIGITS == 10
assert N_CLASSES == 54
