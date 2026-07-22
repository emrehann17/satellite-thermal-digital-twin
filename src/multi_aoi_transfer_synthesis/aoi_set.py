"""Generic AOI-set validation, canonical ordering, and pair/direction
generation for the multi-AOI transfer synthesis.

Nothing in this module knows about any specific AOI name or a fixed region
count. It operates purely on a caller-supplied list of experiment ID strings.
"""
from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass, field

MIN_AOI_COUNT = 2
MAX_AOI_COUNT = 5

# Filesystem-safe slug budget before falling back to a short content hash.
_MAX_SLUG_LEN = 80
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_]+")


class AoiSetError(ValueError):
    """Raised when the caller-supplied AOI selection is invalid."""


@dataclass(frozen=True)
class AoiSet:
    """A validated selection of 2-5 AOI experiment IDs.

    Attributes:
        display_order: AOI IDs in the exact order the caller supplied them
            (used only for human-facing display ordering).
        canonical_order: AOI IDs sorted lexicographically -- used everywhere
            an order-independent, deterministic ordering is required (IDs,
            hashes, output namespace).
        canonical_set_id: a deterministic, filesystem-safe identifier derived
            from `canonical_order`. A readable slug of the sorted AOI IDs,
            or -- if that slug would be too long -- a short content hash of
            the sorted AOI IDs instead.
    """

    display_order: tuple[str, ...]
    canonical_order: tuple[str, ...] = field(init=False)
    canonical_set_id: str = field(init=False)

    def __post_init__(self) -> None:
        canonical_order = tuple(sorted(self.display_order))
        object.__setattr__(self, "canonical_order", canonical_order)
        object.__setattr__(self, "canonical_set_id", _canonical_set_id(canonical_order))

    def ordered_directions(self) -> list[tuple[str, str]]:
        """All N*(N-1) ordered (source, target) directions, source != target,
        generated generically (itertools.permutations) over the canonical
        AOI ordering. Never a hardcoded pair list."""
        return list(itertools.permutations(self.canonical_order, 2))

    def unordered_pairs(self) -> list[tuple[str, str]]:
        """All N-choose-2 unordered pairs, generated generically
        (itertools.combinations) over the canonical AOI ordering. Each pair
        is itself returned in canonical (sorted) order."""
        return list(itertools.combinations(self.canonical_order, 2))


def _canonical_set_id(canonical_order: tuple[str, ...]) -> str:
    joined = "__".join(canonical_order)
    slug = _SAFE_TOKEN_RE.sub("_", joined).strip("_")
    if slug and len(slug) <= _MAX_SLUG_LEN:
        return slug
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    if not slug:
        return f"aoi_set_{digest}"
    # Truncate the readable part and append a short deterministic hash so
    # the id stays both traceable and filesystem-safe.
    budget = _MAX_SLUG_LEN - len(digest) - 1
    truncated = slug[:max(budget, 0)].rstrip("_")
    return f"{truncated}_{digest}" if truncated else f"aoi_set_{digest}"


def canonicalize_pair(experiment_a: str, experiment_b: str) -> tuple[str, str]:
    """Canonicalize an unordered AOI pair the same way AOI sets are
    canonicalized (lexicographic sort), so pair resolution/lookup is
    order-independent."""
    return tuple(sorted((experiment_a, experiment_b)))  # type: ignore[return-value]


def validate_aoi_set(aois: list[str]) -> AoiSet:
    """Validate a caller-supplied AOI experiment-id list and return an
    `AoiSet`.

    Raises:
        AoiSetError: if the count is out of [2, 5], if there are duplicates,
            or if any entry is not a non-empty string.
    """
    if not isinstance(aois, (list, tuple)):
        raise AoiSetError(f"AOI selection must be a list of experiment IDs, got {type(aois)!r}.")

    for entry in aois:
        if not isinstance(entry, str) or not entry.strip():
            raise AoiSetError(f"Each AOI experiment ID must be a non-empty string; got {entry!r}.")

    n = len(aois)
    if n < MIN_AOI_COUNT:
        raise AoiSetError(
            f"At least {MIN_AOI_COUNT} AOI experiment IDs are required; got {n}."
        )
    if n > MAX_AOI_COUNT:
        raise AoiSetError(
            f"At most {MAX_AOI_COUNT} AOI experiment IDs are supported; got {n}."
        )

    seen: dict[str, int] = {}
    for entry in aois:
        seen[entry] = seen.get(entry, 0) + 1
    duplicates = sorted(k for k, v in seen.items() if v > 1)
    if duplicates:
        raise AoiSetError(f"Duplicate AOI experiment IDs are not allowed: {duplicates}.")

    return AoiSet(display_order=tuple(aois))
