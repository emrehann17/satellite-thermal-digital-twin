"""Forbidden-language guard and mandatory Evia framing.

The existing per-AOI guard `window_closure_sensitivity.FORBIDDEN_COMPARE_PHRASES`
is necessary but NOT sufficient: it does not ban `proven`, `causal`, `optimal`,
`best window`, `operationally validated` or `leakage eliminated`. This module
enforces the UNION of that list, the phrases this analysis adds, and the
foreign-factor wording inherited from the compositing counterfactual.

Design reference: docs/multi_region_window_closure_design/SCIENTIFIC_CONTRACT.md 12
"""
from __future__ import annotations

import re
from typing import Any

from src.multi_region_window_closure.contract import (
    DIFFERENT_REGIME_CONTROL_AOI,
    MultiRegionWindowClosureError,
)

#: Phrases added by this analysis that the existing per-AOI guard does not ban.
ADDITIONAL_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "statistically proven",
    "proven",
    "causal",
    "operationally validated",
    "leakage eliminated",
    "optimal",
    "best window",
)

#: Vocabulary that is explicitly permitted and must never be flagged.
PERMITTED_PHRASES: tuple[str, ...] = (
    "bootstrap-supported",
    "interval excludes zero",
    "interval includes zero",
    "uncertainty remains",
    "point estimate",
    "descriptive",
    "direction-dependent",
    "under this frozen design",
)

#: The three phrases that must appear verbatim wherever the different-regime
#: control AOI is characterised.
EVIA_REQUIRED_PHRASES: tuple[str, ...] = (
    "different-regime control",
    "high-prevalence sensitivity region",
    "not an equal-prevalence fourth validation region",
)

#: Framings that would misrepresent the different-regime control.
EVIA_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "fourth validation region",
    "equal-prevalence",
    "equivalent aoi",
)

#: Prose field names walked when scanning a JSON payload. Mirrors
#: `window_closure_sensitivity._PROSE_FIELD_TOKENS`.
PROSE_FIELD_TOKENS: tuple[str, ...] = (
    "note", "notes", "summary", "description", "statement", "conclusion",
    "conclusions", "interpretation", "limitation", "limitations", "reason",
    "wording", "text", "report", "framing", "definition",
)


def _existing_forbidden_phrases() -> tuple[str, ...]:
    """The per-AOI guard's list, imported rather than copied."""
    from src.window_closure_sensitivity import (
        FOREIGN_FACTOR_PHRASES, FORBIDDEN_COMPARE_PHRASES,
    )

    return tuple(FORBIDDEN_COMPARE_PHRASES) + tuple(FOREIGN_FACTOR_PHRASES)


def forbidden_phrases() -> tuple[str, ...]:
    """The UNION enforced by this analysis, deduplicated and ordered."""
    merged = set(_existing_forbidden_phrases()) | set(ADDITIONAL_FORBIDDEN_PHRASES)
    return tuple(sorted(merged))


def _permitted_spans(lowered: str) -> list[tuple[int, int]]:
    """Character spans covered by an explicitly permitted phrase.

    `interval excludes zero` contains no forbidden substring, but permitted
    phrases are masked before scanning anyway so the guard stays correct if
    either list grows.
    """
    spans: list[tuple[int, int]] = []
    for phrase in PERMITTED_PHRASES:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            spans.append((index, index + len(phrase)))
            start = index + 1
    return spans


def find_forbidden(text: str) -> list[str]:
    """Forbidden phrases occurring in `text`, outside any permitted phrase.

    Word-boundary matching prevents `proven` from firing inside `improven`-like
    tokens while still catching it as a standalone claim.
    """
    if not text:
        return []
    lowered = str(text).lower()
    masked = _permitted_spans(lowered)
    hits: list[str] = []
    for phrase in forbidden_phrases():
        pattern = re.compile(
            r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        )
        for match in pattern.finditer(lowered):
            if any(s <= match.start() and match.end() <= e for s, e in masked):
                continue
            hits.append(phrase)
            break
    return sorted(set(hits))


def find_forbidden_locations(
    text: str, *, relative_path: str,
) -> list[dict[str, Any]]:
    """Return fail-closed token evidence with file, line, and short context."""
    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(str(text).splitlines(), start=1):
        for token in find_forbidden(line):
            findings.append({
                "forbidden_token": token,
                "relative_file_path": relative_path,
                "line_number": line_number,
                "context": line.strip()[:240],
            })
    return findings


def collect_prose(payload: Any, in_prose_context: bool = False) -> list[str]:
    """Every prose string in a nested payload.

    Walks dicts and lists, collecting values of prose-named fields plus every
    string inside a prose-named container -- the same strategy as
    `assert_compare_wording`.
    """
    out: list[str] = []
    if isinstance(payload, str):
        if in_prose_context:
            out.append(payload)
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            prose = in_prose_context or any(tok in key_l for tok in PROSE_FIELD_TOKENS)
            out.extend(collect_prose(value, prose))
        return out
    if isinstance(payload, (list, tuple)):
        for item in payload:
            out.extend(collect_prose(item, in_prose_context))
    return out


def assert_no_forbidden_language(payload: Any, where: str) -> None:
    """Fail closed on prohibited significance, causal or optimality language."""
    texts = [payload] if isinstance(payload, str) else collect_prose(payload)
    offenders: dict[str, list[str]] = {}
    for text in texts:
        hits = find_forbidden(text)
        if hits:
            offenders.setdefault(text[:120], []).extend(hits)
    if offenders:
        first = next(iter(offenders.items()))
        raise MultiRegionWindowClosureError(
            f"BLOCKER: FORBIDDEN_LANGUAGE -- {where} contains "
            f"{sorted(set(first[1]))} in: {first[0]!r}"
        )


def assert_evia_framing(text: str, where: str) -> None:
    """The different-regime control must be framed correctly wherever named."""
    lowered = str(text).lower()
    if DIFFERENT_REGIME_CONTROL_AOI not in lowered:
        return
    missing = [p for p in EVIA_REQUIRED_PHRASES if p not in lowered]
    if missing:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: EVIA_FRAMING_MISSING -- {where} names "
            f"{DIFFERENT_REGIME_CONTROL_AOI} without {missing}."
        )
    present = [p for p in EVIA_FORBIDDEN_PHRASES if p in lowered]
    # "not an equal-prevalence fourth validation region" legitimately contains
    # two of the forbidden fragments, so they are only violations when they
    # appear OUTSIDE that mandated negation.
    negation = EVIA_REQUIRED_PHRASES[2]
    without_negation = lowered.replace(negation, " ")
    real = [p for p in present if p in without_negation]
    if real:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: EVIA_FRAMING_VIOLATION -- {where} describes "
            f"{DIFFERENT_REGIME_CONTROL_AOI} as {real}."
        )


def evia_regime_note() -> str:
    """The canonical framing sentence, used wherever the control is summarised."""
    return (
        "evia_2021_extended is a different-regime control and a "
        "high-prevalence sensitivity region; it is not an equal-prevalence "
        "fourth validation region. Its PR-AUC and Brier levels are not "
        "comparable across AOIs."
    )


def assert_wording_contract(payload: Any, where: str) -> None:
    """Both guards in one call: forbidden language and Evia framing."""
    assert_no_forbidden_language(payload, where)
    texts = [payload] if isinstance(payload, str) else collect_prose(payload)
    for text in texts:
        assert_evia_framing(text, where)
