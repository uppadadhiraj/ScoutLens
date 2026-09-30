"""Citation validation for LLM-written prose: NO SOURCE = NO FACT, enforced in code.

The LLM is shown numbered evidence items and must end each sentence with ``[E#]`` markers. Any
sentence with no marker, or with a marker that is not in the evidence store, is dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from services.evidence.store import EvidenceStore

_CITATION = re.compile(r"\[(E\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
# "X. [E1] Y" -> "X [E1]. Y": models cite before or after the full stop; normalise to "before".
_TRAILING_CITATIONS = re.compile(r"([.!?])\s*((?:\[E\d+\]\s*)+)")


def _sentences(text: str) -> list[str]:
    text = _TRAILING_CITATIONS.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", " ".join(text.split()))
    return [s.strip() for s in _SENTENCE_SPLIT.split(" ".join(text.split())) if s.strip()]


@dataclass
class ValidatedProse:
    text: str  # kept sentences, citations preserved
    evidence_ids: list[str]  # distinct ids cited by the kept sentences, in order
    dropped: int  # sentences removed for missing/invalid citations

    @property
    def is_empty(self) -> bool:
        return not self.text


def cited_ids(text: str) -> list[str]:
    return list(dict.fromkeys(_CITATION.findall(text)))


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[E\d+\]", "", text).strip()


def validate_prose(text: str, store: EvidenceStore) -> ValidatedProse:
    """Keep only sentences whose citations all resolve to real evidence."""
    kept: list[str] = []
    ids: list[str] = []
    dropped = 0
    for sentence in _sentences(text):
        cited = cited_ids(sentence)
        if cited and len(store.existing_ids(cited)) == len(cited):
            kept.append(sentence)
            ids += [i for i in cited if i not in ids]
        else:
            dropped += 1
    return ValidatedProse(text=" ".join(kept), evidence_ids=ids, dropped=dropped)
