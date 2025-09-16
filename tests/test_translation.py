from echoliner.translation import ParallelCorpus, StatisticalTranslator


import numpy as np

from echoliner.translation import (
    DomainLexicon,
    DynamicTimeWarping,
    ParallelCorpus,
    SpeechFeatureExtractor,
    StatisticalTranslator,
    alignment_path,
    bleu_score,
    chrf_score,
    griffin_lim,
    soft_alignment,
    translation_error_rate,
)


def _build_translator() -> StatisticalTranslator:
    corpus = ParallelCorpus(
        pairs=[
            ("hello automation", "你好 自动化"),
            ("smart control", "智能 控制"),
            ("robot assembly", "机器人 装配"),
            ("precision analytics", "精准 分析"),
        ]
    )
    return StatisticalTranslator(corpus, smoothing=0.2)


def test_translate_forward_direction() -> None:
    translator = _build_translator()
    output = translator.translate("robot assembly", source_lang="en", target_lang="zh")
    assert output == "机器人装配"


def test_translate_backward_direction() -> None:
    translator = _build_translator()
    output = translator.translate("你好 自动化", source_lang="zh", target_lang="en")
    assert output == "hello automation"


def test_alignment_matrix_highlights_diagonal() -> None:
    translator = _build_translator()
    matrix = translator.alignment_matrix("hello automation", "你好 自动化", direction="forward")
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] > matrix[0, 1]
    assert matrix[1, 1] > matrix[1, 0]


def test_adaptation_updates_vocab() -> None:
    translator = _build_translator()
    baseline = translator.translate("cobot", source_lang="en", target_lang="zh")
    assert baseline == "cobot"
    translator.adapt("cobot", "协作 机器人")
    updated = translator.translate("cobot", source_lang="en", target_lang="zh")
    assert updated == "协作机器人"


def test_domain_lexicon_matches_phrase() -> None:
    lexicon = DomainLexicon()
    entry = lexicon.lookup("adaptive actuator alignment")
    assert entry is not None
    segments = lexicon.greedy_lookup("adaptive actuator alignment".split())
    assert segments[0][1]


def test_speech_feature_extractor_shapes() -> None:
    extractor = SpeechFeatureExtractor()
    waveform = np.zeros(extractor.sample_rate)
    mel = extractor.mel_spectrogram(waveform)
    assert mel.shape[0] == extractor.n_mels


def test_griffin_lim_returns_waveform() -> None:
    magnitude = np.abs(np.random.randn(257, 10))
    waveform = griffin_lim(magnitude, n_iter=5)
    assert waveform.ndim == 1


def test_alignment_utilities() -> None:
    embeddings_a = np.eye(3)
    embeddings_b = np.eye(3)
    dtw = DynamicTimeWarping(lambda a, b: float(np.linalg.norm(a - b)))
    cost, path = dtw.compute(embeddings_a, embeddings_b)
    assert np.isclose(cost, 0.0)
    assert len(path) == 3
    soft = soft_alignment(embeddings_a, embeddings_b)
    assert np.allclose(soft.sum(axis=1), 1.0)


def test_translation_metrics_behave() -> None:
    reference = "智能 控制".split()
    hypothesis = "智能 控制".split()
    assert bleu_score(reference, hypothesis) > 0.9
    assert translation_error_rate(reference, hypothesis) == 0.0
    assert chrf_score("你好自动化", "你好自动化") == 1.0
