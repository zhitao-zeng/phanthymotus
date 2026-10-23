"""Long numeric dictation is rendered without modifying prose or model names."""

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401
from plugins.numeric_dictation import normalize_long_numeric_dictation


@pytest.mark.parametrize('text, language, expected', [
    ('zero seven two six double one nine triple eight.', 'en', '0726119888'),
    ('Seven billion two hundred and sixty million nine hundred eighteen thousand forty two.', 'en', '7260918042'),
    ('one million and seventeen', 'en', '1000017'),
    ('七二六零九一八零四二。', 'zh', '7260918042'),
    ('幺〇三四五六七八九', 'zh', '103456789'),
])
def test_long_numeric_only_dictation(text, language, expected):
    assert normalize_long_numeric_dictation(text, language) == expected


@pytest.mark.parametrize('text', [
    'We tested Whisper large-v3 and F5-TTS.',
    'One million people use this service.',
    'The number is zero seven two six one one nine eight eight eight.',
    'one', 'oh', 'twenty three', 'one two hundred million',
    'one million million', 'one million and',
])
def test_prose_short_answers_and_invalid_numbers_stay_unchanged(text):
    assert normalize_long_numeric_dictation(text, 'en') == text


def test_unsupported_language_and_chinese_prose_stay_unchanged():
    text = 'one billion two hundred million'
    assert normalize_long_numeric_dictation(text, 'fr') == text
    text = '我们比较了一百万条录音。'
    assert normalize_long_numeric_dictation(text, 'zh') == text
