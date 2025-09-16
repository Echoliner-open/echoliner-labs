"""Statistical word-to-word translator for bilingual manufacturing lexicons."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .lexicon import DomainLexicon

__all__ = ["ParallelCorpus", "StatisticalTranslator"]

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PUNCTUATION = {",", ".", ";", ":", "!", "?", "。", "，", "！", "？"}


def _tokenize(text: str) -> List[str]:
    tokens = []
    for match in _TOKEN_PATTERN.findall(text):
        token = match.lower()
        tokens.append(token)
    return tokens


def _detokenize(tokens: Sequence[str]) -> str:
    output: List[str] = []
    for token in tokens:
        if not output:
            output.append(token)
            continue
        if token in _PUNCTUATION or not token.isascii():
            output.append(token)
        else:
            output.append(" " + token)
    return "".join(output)


def _combine_tokens(tokens: Sequence[str]) -> str:
    if all(tok.isascii() for tok in tokens):
        return " ".join(tokens)
    return "".join(tokens)


@dataclass(frozen=True)
class ParallelCorpus:
    """Parallel sentences curated for the manufacturing domain."""

    pairs: Sequence[Tuple[str, str]]
    source_lang: str = "en"
    target_lang: str = "zh"

    def tokenized(self) -> List[Tuple[List[str], List[str]]]:
        return [(_tokenize(src), _tokenize(tgt)) for src, tgt in self.pairs]


class StatisticalTranslator:
    """Symmetric IBM Model-1 style translator with Laplace smoothing."""

    def __init__(
        self,
        corpus: ParallelCorpus,
        smoothing: float = 0.5,
        lexicon: DomainLexicon | None = None,
    ):
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self._corpus = corpus
        self._smoothing = smoothing
        self._lexicon = lexicon or DomainLexicon()
        token_pairs = corpus.tokenized()
        self._forward_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self._backward_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self._forward_vocab: set[str] = set()
        self._backward_vocab: set[str] = set()
        self._forward_memory: Dict[str, str] = {}
        self._backward_memory: Dict[str, str] = {}
        for src_tokens, tgt_tokens in token_pairs:
            self._forward_vocab.update(tgt_tokens)
            self._backward_vocab.update(src_tokens)
            self._accumulate(self._forward_counts, src_tokens, tgt_tokens)
            self._accumulate(self._backward_counts, tgt_tokens, src_tokens)
            self._update_phrase_memory(self._forward_memory, src_tokens, tgt_tokens)
            self._update_phrase_memory(self._backward_memory, tgt_tokens, src_tokens)

    def _accumulate(
        self,
        table: DefaultDict[str, Counter[str]],
        sources: Iterable[str],
        targets: Iterable[str],
    ) -> None:
        source_list = list(sources)
        target_list = list(targets)
        for source in source_list:
            counter = table[source]
            for target in target_list:
                counter[target] += 1
        for source_token, target_token in zip(source_list, target_list):
            table[source_token][target_token] += 2

    def _update_phrase_memory(
        self,
        memory: Dict[str, str],
        sources: Sequence[str],
        targets: Sequence[str],
    ) -> None:
        if len(sources) == 1 and len(targets) > 1:
            memory[sources[0]] = _combine_tokens(targets)

    def adapt(self, source_text: str, target_text: str) -> None:
        src_tokens = _tokenize(source_text)
        tgt_tokens = _tokenize(target_text)
        self._forward_vocab.update(tgt_tokens)
        self._backward_vocab.update(src_tokens)
        self._accumulate(self._forward_counts, src_tokens, tgt_tokens)
        self._accumulate(self._backward_counts, tgt_tokens, src_tokens)
        self._update_phrase_memory(self._forward_memory, src_tokens, tgt_tokens)
        self._update_phrase_memory(self._backward_memory, tgt_tokens, src_tokens)

    def _conditional_probability(
        self,
        counts: Mapping[str, int],
        vocab_size: int,
        token: str,
    ) -> float:
        denominator = sum(counts.values()) + self._smoothing * (vocab_size + 1)
        value = counts.get(token, 0)
        if token in counts:
            numerator = value + self._smoothing
        else:
            numerator = self._smoothing
        return float(numerator / denominator)

    def _best_translation(
        self,
        counts_table: Mapping[str, Counter[str]],
        vocab_size: int,
        token: str,
    ) -> str:
        counts = counts_table.get(token)
        if not counts:
            return token
        best_target = max(
            counts.items(),
            key=lambda item: self._conditional_probability(counts, vocab_size, item[0]),
        )[0]
        return best_target

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        direction = self._resolve_direction(source_lang, target_lang)
        tokens = _tokenize(text)
        segments = self._lexicon.greedy_lookup(tokens)
        if direction == "forward":
            vocab_size = len(self._forward_vocab)
            table = self._forward_counts
            memory = self._forward_memory
        else:
            vocab_size = len(self._backward_vocab)
            table = self._backward_counts
            memory = self._backward_memory
        translated_tokens: List[str] = []
        for segment, matched in segments:
            if matched:
                translated_tokens.append(segment)
                continue
            token = segment
            if token in memory:
                translated_tokens.append(memory[token])
            else:
                translated_tokens.append(self._best_translation(table, vocab_size, token))
        return _detokenize(translated_tokens)

    def alignment_matrix(
        self,
        source: Sequence[str] | str,
        target: Sequence[str] | str,
        *,
        direction: str = "forward",
    ) -> np.ndarray:
        if isinstance(source, str):
            source_tokens = _tokenize(source)
        else:
            source_tokens = [tok.lower() for tok in source]
        if isinstance(target, str):
            target_tokens = _tokenize(target)
        else:
            target_tokens = [tok.lower() for tok in target]

        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'")
        if direction == "forward":
            table = self._forward_counts
            vocab_size = len(self._forward_vocab)
        else:
            table = self._backward_counts
            vocab_size = len(self._backward_vocab)

        matrix = np.zeros((len(source_tokens), len(target_tokens)), dtype=float)
        for i, src_token in enumerate(source_tokens):
            counts = table.get(src_token, {})
            for j, tgt_token in enumerate(target_tokens):
                matrix[i, j] = self._conditional_probability(counts, vocab_size, tgt_token)
        return matrix

    def _resolve_direction(self, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            raise ValueError("Source and target languages must differ")
        if (source_lang, target_lang) == (
            self._corpus.source_lang,
            self._corpus.target_lang,
        ):
            return "forward"
        if (source_lang, target_lang) == (
            self._corpus.target_lang,
            self._corpus.source_lang,
        ):
            return "backward"
        raise ValueError("Language pair not supported by this translator")
