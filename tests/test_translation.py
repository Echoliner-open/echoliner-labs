from echoliner.translation import ParallelCorpus, StatisticalTranslator


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
