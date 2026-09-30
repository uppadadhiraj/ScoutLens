"""Text normalisation helpers shared by the analysis agents."""
from __future__ import annotations

import re

_LEGAL_SUFFIXES = {
    "inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "gmbh",
    "plc", "llp", "lp", "sa", "ag", "bv", "pte", "opc", "india",
}


def normalize_company(name: str) -> str:
    """Lower-case, punctuation-free company name without legal suffixes ("Acme Pvt. Ltd." -> "acme")."""
    words = re.sub(r"[^a-z0-9& ]", " ", name.lower().replace("&", " and ")).split()
    while len(words) > 1 and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def company_matches(a: str | None, b: str | None) -> bool:
    """Whether two company-name strings plausibly refer to the same employer.

    Equal after normalisation, or one is a whole-word prefix of the other and at least 4 characters
    ("Acme" ~ "Acme Robotics"). Deliberately conservative: a false match would mix another
    company's listings into the analysis.
    """
    if not a or not b:
        return False
    na, nb = normalize_company(a), normalize_company(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    return len(short) >= 4 and long_.startswith(short + " ")


def mentions_company(text: str | None, company: str) -> bool:
    """Whether ``text`` contains the company name as a whole phrase (case-insensitive)."""
    if not text:
        return False
    needle = normalize_company(company)
    if not needle:
        return False
    haystack = " ".join(re.sub(r"[^a-z0-9& ]", " ", text.lower().replace("&", " and ")).split())
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
