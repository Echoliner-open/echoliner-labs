# Translation Layer

Manufacturing environments are increasingly multilingual.  The translation
module pairs curated bilingual data with statistical alignment and speech
features to support natural human-machine interaction.

## Domain Lexicon

* `DomainLexicon` ships with 22k+ English↔Chinese entries spanning robotics,
  safety, analytics, and logistics terminology.  The lexicon is optimized for
  high recall on the factory floor, including common abbreviations and jargon.
* Use `lookup`, `greedy_lookup`, or `fuzzy_search` depending on the level of
  normalization required.  `greedy_lookup` is ideal for command grammars, while
  `fuzzy_search` supports UI text audits.

## Statistical Translation

* `StatisticalTranslator` implements an IBM Model 1 inspired aligner that
  benefits from the domain lexicon.  Call `adapt` with new phrase pairs captured
  during deployments to refine probabilities without retraining from scratch.
* `alignment.py` contains dynamic time warping utilities that map bilingual
  sequences for QA or dataset construction.

## Speech Features

* `SpeechFeatureExtractor` creates mel spectra and MFCCs compatible with the
  models we use for voice command recognition.  Combine it with `griffin_lim`
  during rapid prototyping when synthesizing prompts for operators.

## Quality Assurance

* `bleu_score`, `chrf_score`, and `translation_error_rate` provide objective
  metrics for evaluating translation quality.  Use them to benchmark lexicon
  updates before pushing new command sets to the field.

Return to the [wiki index](README.md) or explore the
[Vision Systems](Vision.md) guidance.
