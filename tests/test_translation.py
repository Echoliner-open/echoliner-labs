from echoliner.translation.simple_translator import translate


def test_translate_en_to_zh_known_word():
    assert translate("robot", "en", "zh") == "机器人"


def test_translate_unknown_word_returns_input():
    assert translate("unknown", "en", "zh") == "unknown"


def test_translate_zh_to_en():
    assert translate("工厂", "zh", "en") == "factory"
