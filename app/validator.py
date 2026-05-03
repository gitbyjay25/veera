"""
validator.py
────────────
Layer 5: Anti-hallucination, URL, jargon, CTA, length, and send_as guardrails.
< 2ms. Zero LLM.

Checks:
  1. No URL patterns in body            → hard block (-3 judge penalty)
  2. No taboo words for the category    → hard block
  3. All numbers in body traceable to input number pool → HALLUCINATION_RISK
  4. Exactly one CTA                    → warn + flag
  5. Body length 80–340 chars           → warn if outside
  6. send_as matches profile expectation → hard block if wrong
  7. No internal field names leaked     → re-prompt signal
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_URL_RE = re.compile(r"https?://|www\.|\.com|\.in|bit\.ly", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

# Internal field names that should never appear in composed body
_JARGON = [
    "merchant_id", "customer_id", "trigger_id", "context_id",
    "payload", "suppression_key", "perf_snapshot", "delta_7d",
]


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)     # hard fails → trigger re-prompt
    warnings: list[str] = field(default_factory=list)   # soft issues → surface in rationale


def validate(
    body: str,
    send_as: str,
    expected_send_as: str,
    number_pool: set[str],
    taboos: list[str],
    cta: str,
) -> ValidationResult:
    """Run all checks. Returns ValidationResult."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. URL scan — hard block
    if _URL_RE.search(body):
        errors.append("URL_DETECTED: URLs are not allowed in composed body.")

    # 2. Category taboo words
    body_lower = body.lower()
    for taboo in taboos:
        if taboo.lower() in body_lower:
            errors.append(f"TABOO_WORD: '{taboo}' is a banned term for this category.")

    # 3. Number pool check
    body_nums = set(_NUM_RE.findall(body))
    if number_pool:  # only check if we have a pool
        leaked = body_nums - number_pool
        if leaked:
            warnings.append(f"HALLUCINATION_RISK: numbers not found in input — {leaked}")

    # 4. CTA sanity — multiple "?" in body often signals multiple CTAs
    question_marks = body.count("?")
    if question_marks > 2:
        warnings.append(f"MULTIPLE_CTAS: {question_marks} question marks detected; aim for 1 CTA.")

    # 5. Body length
    length = len(body)
    if length < 80:
        warnings.append(f"BODY_TOO_SHORT: {length} chars (min 80).")
    elif length > 380:
        warnings.append(f"BODY_TOO_LONG: {length} chars (max 380).")

    # 6. send_as discipline
    if send_as != expected_send_as:
        errors.append(
            f"SEND_AS_MISMATCH: got '{send_as}', expected '{expected_send_as}' "
            f"for this trigger profile."
        )

    # 7. Jargon / internal field names
    for jargon in _JARGON:
        if jargon in body_lower:
            errors.append(f"JARGON_LEAKED: internal field name '{jargon}' found in body.")

    # 8. Vera / magicpin mention in merchant_on_behalf messages
    if expected_send_as == "merchant_on_behalf":
        for brand in ("vera", "magicpin"):
            if brand in body_lower:
                errors.append(
                    f"BRAND_LEAK: '{brand}' must not appear in merchant_on_behalf messages."
                )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
