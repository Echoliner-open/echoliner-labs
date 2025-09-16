"""Simple offline translation utilities.

This is a placeholder demonstrating how EchoLiner might expose
translation capabilities without relying on external services.
"""

from __future__ import annotations

from typing import Dict

# Minimal toy dictionaries for demonstration purposes.
_EN_TO_ZH: Dict[str, str] = {
    "hello": "你好",
    "robot": "机器人",
    "factory": "工厂",
    "automation": "自动化",
}

_ZH_TO_EN: Dict[str, str] = {v: k for k, v in _EN_TO_ZH.items()}


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between English and Chinese.

    Parameters
    ----------
    text: str
        Input text (single word) to translate.
    source_lang: str
        Source language code ('en' or 'zh').
    target_lang: str
        Target language code ('en' or 'zh').

    Returns
    -------
    str
        Translated word. Unknown words return the input text unchanged.
    """
    if source_lang == target_lang:
        return text

    if source_lang == "en" and target_lang == "zh":
        return _EN_TO_ZH.get(text.lower(), text)
    if source_lang == "zh" and target_lang == "en":
        return _ZH_TO_EN.get(text, text)

    raise ValueError("Unsupported language pair")
