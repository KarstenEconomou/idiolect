"""Find normalized training-text matches with bounded memory growth."""

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

_HASH_BASE = 257
_HASH_MASK = (1 << 64) - 1
_MAX_SEED_CHARS = 16


@dataclass(frozen=True, slots=True)
class TrainingMatchIndex:
    """Keep sparse training-text seeds for exact match verification."""

    threshold: int
    seed_chars: int
    normalized: tuple[str, ...]
    exact: frozenset[str]
    seeds: Mapping[int, tuple[int, ...]]

    @classmethod
    def build(
        cls, texts: Sequence[str], threshold: int
    ) -> TrainingMatchIndex:
        """Build one sparse index for a minimum match length."""
        normalized = tuple(normalize_text(text) for text in texts)
        seed_chars = min(threshold, _MAX_SEED_CHARS)
        stride = threshold - seed_chars + 1
        values: dict[int, set[int]] = {}
        for text_index, text in enumerate(normalized):
            if len(text) < threshold:
                continue
            hashes = _rolling_hashes(text, seed_chars)
            for start in range(0, len(hashes), stride):
                values.setdefault(hashes[start], set()).add(text_index)
        return cls(
            threshold,
            seed_chars,
            normalized,
            frozenset(value for value in normalized if value),
            {
                key: tuple(sorted(indices))
                for key, indices in values.items()
            },
        )

    def longest(self, candidate: str) -> int:
        """Return the longest verified match that can meet the threshold."""
        normalized = normalize_text(candidate)
        if len(normalized) < self.threshold:
            return 0
        candidates: set[int] = set()
        for value in _rolling_hashes(normalized, self.seed_chars):
            candidates.update(self.seeds.get(value, ()))
        best = 0
        for text_index in candidates:
            match = SequenceMatcher(
                None,
                normalized,
                self.normalized[text_index],
                autojunk=False,
            ).find_longest_match()
            best = max(best, match.size)
        return best


def normalize_text(text: str) -> str:
    """Return text in the stable comparison form."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _rolling_hashes(text: str, size: int) -> tuple[int, ...]:
    if len(text) < size:
        return ()
    high = pow(_HASH_BASE, size - 1, _HASH_MASK + 1)
    value = 0
    for character in text[:size]:
        value = ((value * _HASH_BASE) + ord(character)) & _HASH_MASK
    result = [value]
    for start in range(1, len(text) - size + 1):
        value = (
            (value - (ord(text[start - 1]) * high)) * _HASH_BASE
            + ord(text[start + size - 1])
        ) & _HASH_MASK
        result.append(value)
    return tuple(result)
