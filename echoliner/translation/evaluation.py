"""Evaluation metrics for translation quality."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence
import math

__all__ = ["bleu_score", "chrf_score", "translation_error_rate"]


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: Sequence[str], hypothesis: Sequence[str], *, max_order: int = 4, smoothing: float = 1.0) -> float:
    ref_length = len(reference)
    hyp_length = len(hypothesis)
    if hyp_length == 0:
        return 0.0
    log_precisions: List[float] = []
    for order in range(1, max_order + 1):
        ref_ngrams = _ngrams(reference, order)
        hyp_ngrams = _ngrams(hypothesis, order)
        possible = len(hypothesis) - order + 1
        if possible <= 0:
            continue
        overlap = hyp_ngrams & ref_ngrams
        match = sum(overlap.values())
        precision = (match + smoothing) / (possible + smoothing)
        log_precisions.append(math.log(precision))
    if not log_precisions:
        return 0.0
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    brevity = 1.0
    if hyp_length < ref_length and ref_length > 0:
        brevity = math.exp(1 - ref_length / hyp_length)
    return brevity * geo_mean


def translation_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    insertions = deletions = substitutions = 0
    i = j = 0
    while i < len(reference) and j < len(hypothesis):
        if reference[i] == hypothesis[j]:
            i += 1
            j += 1
        else:
            substitutions += 1
            i += 1
            j += 1
    deletions += len(reference) - i
    insertions += len(hypothesis) - j
    total = len(reference)
    if total == 0:
        return 0.0
    return (insertions + deletions + substitutions) / total


def chrf_score(reference: str, hypothesis: str, *, beta: float = 2.0, n: int = 6) -> float:
    if reference == hypothesis:
        return 1.0
    def char_ngrams(text: str, order: int) -> Counter[str]:
        return Counter(text[i : i + order] for i in range(len(text) - order + 1))

    precisions: List[float] = []
    recalls: List[float] = []
    for order in range(1, n + 1):
        ref_ngrams = char_ngrams(reference, order)
        hyp_ngrams = char_ngrams(hypothesis, order)
        overlap = ref_ngrams & hyp_ngrams
        match = sum(overlap.values())
        ref_total = sum(ref_ngrams.values())
        hyp_total = sum(hyp_ngrams.values())
        precisions.append(match / max(hyp_total, 1))
        recalls.append(match / max(ref_total, 1))
    precision = sum(precisions) / n
    recall = sum(recalls) / n
    if precision + recall == 0:
        return 0.0
    beta_sq = beta**2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
