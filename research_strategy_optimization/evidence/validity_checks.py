"""Independent protocol/manifest checks used by the trusted verifier."""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence, Tuple


def check_hashes(manifest: Mapping[str, object], expected: Mapping[str, object]) -> Tuple[bool, Tuple[str, ...]]:
    reasons = []
    for key, value in expected.items():
        if manifest.get(key) != value:
            reasons.append(f"{key}_mismatch")
    return not reasons, tuple(reasons)


def check_independent_seeds(seeds: Sequence[int], forbidden: Iterable[int] = ()) -> Tuple[bool, Tuple[str, ...]]:
    values = tuple(int(s) for s in seeds)
    reasons = []
    if not values:
        reasons.append("empty_seed_set")
    if len(set(values)) != len(values):
        reasons.append("duplicate_seed")
    overlap = set(values).intersection(int(s) for s in forbidden)
    if overlap:
        reasons.append("exploration_confirmation_seed_overlap")
    return not reasons, tuple(reasons)


def scan_for_hidden_identifiers(payload: object, forbidden_tokens: Sequence[str]) -> Tuple[bool, Tuple[str, ...]]:
    text = repr(payload).lower()
    hits = tuple(token for token in forbidden_tokens if token.lower() in text)
    return not hits, hits

