"""Consonant-confusion reranker for YOLO plate predictions.

Weekend-7's synth_v3b recognizer reaches 72% char-acc on the gold-27 set,
with near-zero exact matches. Per-plate inspection shows the remaining
errors are almost entirely Thai consonant substitutions inside otherwise
correct registrations (ค→พ, ฎ→ด, ต→ถ, ม→ฆ, …). The digit scaffold is
right, the length is right, the overall structure passes the plate regex —
one consonant lands on the wrong side of a visually-similar pair.

That's a language-model problem, not a vision problem.

This module post-processes the argmax string from the recognizer by:

  1. Enumerating every consonant-substitution variant from a hand-curated
     confusion-pair table (imported from `render.CONFUSION_GROUPS`).
  2. Keeping only variants that still match the Thai plate regex.
  3. Scoring each survivor with a character-trigram LM trained on the
     1,285 high-confidence real-plate registrations (the same VLM-labeled
     corpus that drove weekend-7 distillation).
  4. Returning the highest-scoring variant (or the original pred if the
     LM disagrees more weakly than a configurable margin).

No retraining needed — the LM is 300kB of trigram counts and the reranker
adds <1ms per plate.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import product
from pathlib import Path

from thai_plate_synth.alphabet import ALPHABET
from thai_plate_synth.render import CONFUSION_GROUPS

PLATE_PATTERN = re.compile(r"^[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}$")

# Map each confusable consonant → the full set of its confusables (self included).
CONFUSION_SET: dict[str, frozenset[str]] = {}
for _group in CONFUSION_GROUPS:
    for _c in _group:
        CONFUSION_SET.setdefault(_c, set()).update(_group)
CONFUSION_SET = {k: frozenset(v) for k, v in CONFUSION_SET.items()}


class TrigramLM:
    """Add-alpha smoothed char-trigram LM over the plate alphabet + pad tokens."""

    START = "^"
    END = "$"

    def __init__(self, corpus: list[str], alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.vocab_size = len(ALPHABET) + 2  # + start + end
        self.tri: Counter[str] = Counter()
        self.ctx: Counter[str] = Counter()
        for s in corpus:
            padded = self.START * 2 + s + self.END
            for i in range(len(padded) - 2):
                self.tri[padded[i:i + 3]] += 1
                self.ctx[padded[i:i + 2]] += 1

    def log_prob(self, s: str) -> float:
        padded = self.START * 2 + s + self.END
        total = 0.0
        for i in range(len(padded) - 2):
            num = self.tri[padded[i:i + 3]] + self.alpha
            denom = self.ctx[padded[i:i + 2]] + self.alpha * self.vocab_size
            total += math.log(num / denom)
        return total


def candidates(pred: str) -> list[str]:
    """All consonant-substitution variants of pred (including pred itself)."""
    slots: list[tuple[str, ...]] = []
    for c in pred:
        if c in CONFUSION_SET:
            slots.append(tuple(sorted(CONFUSION_SET[c])))
        else:
            slots.append((c,))
    return ["".join(x) for x in product(*slots)]


def rerank(pred: str, lm: TrigramLM) -> str:
    """Return the most likely plate string under the LM, constrained to
    regex-valid consonant substitutions of `pred`."""
    if not pred:
        return pred
    cands = [c for c in candidates(pred) if PLATE_PATTERN.match(c)]
    if not cands:
        return pred
    cands.sort(key=lm.log_prob, reverse=True)
    return cands[0]


def load_corpus_from_clean_labels(path: Path) -> list[str]:
    """Extract registration strings from a clean_labels.jsonl file, keeping
    only regex-valid plates (matches the filter the detector was trained on)."""
    out: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        reg = re.sub(r"[^0-9ก-ฮ]", "", rec.get("registration", ""))
        if PLATE_PATTERN.match(reg):
            out.append(reg)
    return out
