"""Utilities for operating on the large domain lexicon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from .domain_lexicon import DOMAIN_LEXICON

__all__ = ["DomainLexicon", "LexiconEntry"]


@dataclass(frozen=True)
class LexiconEntry:
    """Representation of a bilingual lexicon record."""

    source: str
    target: str
    context: str

    @property
    def source_tokens(self) -> List[str]:
        return self.source.split()


class DomainLexicon:
    """Index for the 20k+ entry bilingual lexicon."""

    def __init__(self, entries: Sequence[tuple[str, str, str]] | None = None):
        dataset = entries or DOMAIN_LEXICON
        self._entries: List[LexiconEntry] = [LexiconEntry(*row) for row in dataset]
        self._max_tokens = 0
        self._lookup: dict[str, LexiconEntry] = {}
        self._by_context: dict[str, List[LexiconEntry]] = {}
        for entry in self._entries:
            key = entry.source.lower()
            self._lookup[key] = entry
            self._max_tokens = max(self._max_tokens, len(entry.source_tokens))
            self._by_context.setdefault(entry.context, []).append(entry)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def lookup(self, phrase: str, *, context: str | None = None) -> LexiconEntry | None:
        entry = self._lookup.get(phrase.lower())
        if entry is None:
            return None
        if context is None or entry.context == context:
            return entry
        return None

    def contextual_candidates(self, *, context: str) -> Sequence[LexiconEntry]:
        return self._by_context.get(context, [])

    def greedy_lookup(
        self, tokens: Sequence[str], *, context: str | None = None
    ) -> List[tuple[str, bool]]:
        results: List[tuple[str, bool]] = []
        idx = 0
        while idx < len(tokens):
            match_length = 0
            match_entry: LexiconEntry | None = None
            for window in range(min(self._max_tokens, len(tokens) - idx), 0, -1):
                candidate = " ".join(tokens[idx : idx + window])
                entry = self.lookup(candidate, context=context)
                if entry is not None:
                    match_length = window
                    match_entry = entry
                    break
            if match_entry is None:
                results.append((tokens[idx], False))
                idx += 1
            else:
                results.append((match_entry.target, True))
                idx += match_length
        return results

    def greedy_translate(self, tokens: Sequence[str], *, context: str | None = None) -> List[str]:
        return [text for text, _ in self.greedy_lookup(tokens, context=context)]

    def fuzzy_search(self, query: str, *, top_k: int = 5) -> List[LexiconEntry]:
        scored: List[tuple[float, LexiconEntry]] = []
        query_tokens = query.lower().split()
        for entry in self._entries:
            overlap = self._token_overlap(query_tokens, entry.source_tokens)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    @staticmethod
    def _token_overlap(a: Sequence[str], b: Sequence[str]) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union
