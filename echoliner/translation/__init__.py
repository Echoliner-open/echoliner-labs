"""Translation utilities for cross-language factory orchestration."""

from .alignment import DynamicTimeWarping, alignment_path, soft_alignment
from .evaluation import bleu_score, chrf_score, translation_error_rate
from .lexicon import DomainLexicon
from .simple_translator import ParallelCorpus, StatisticalTranslator
from .speech import SpeechFeatureExtractor, griffin_lim

__all__ = [
    "ParallelCorpus",
    "StatisticalTranslator",
    "DomainLexicon",
    "SpeechFeatureExtractor",
    "griffin_lim",
    "DynamicTimeWarping",
    "alignment_path",
    "soft_alignment",
    "bleu_score",
    "chrf_score",
    "translation_error_rate",
]
