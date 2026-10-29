"""Number rendering must preserve meaning and nonnumeric content."""

import os

import pytest

from plugins.asr_numeric import NumericOutputNormalizer


@pytest.fixture
def normalizer():
    root = os.environ.get("ASR_NUMERIC_GRAMMAR_DIR")
    if not root:
        pytest.skip("Set ASR_NUMERIC_GRAMMAR_DIR for native grammar integration")
    return NumericOutputNormalizer(root, ("GPT-4.1", "ElevenLabs", "Whisper large-v3", "4Paradigm"))


@pytest.mark.parametrize("language,source,expected", [
    ("en", "One of the Eleven Labs models uses twenty percent.", "One of the Eleven Labs models uses 20%."),
    ("en", "OpenAI measured minus three point five volts.", "OpenAI measured -3.5 volts."),
    ("en", "It started in twenty fifteen.", "It started in 2015."),
    ("en", "zero zero one two three four five six seven", "001234567"),
    ("en", "零一八六一零二二三三四五。", "018610223345"),
    ("en", "18,610,223,345.", "18610223345"),
    ("en", "18,600,223,345.", "18600223345"),
    ("en", "GPT four point one and Whisper large v three", "GPT-4.1 and Whisper large-v3"),
    ("en", "GPT four point two", "GPT 4.2"),
    ("en", "4Paradigm model-v2 user02@example.com https://example.com/v3", "4Paradigm model-v2 user02@example.com https://example.com/v3"),
    ("zh", "公司成立于二零一五年，比例百分之九十三点二。", "公司成立于2015年，比例93.2%。"),
    ("zh", "电压负三点六伏，一起开始。", "电压-3.6伏，一起开始。"),
    ("ja", "二千二十五年、九十三パーセントです。", "2025年、93%です。"),
    ("fr", "Un modèle a vingt cinq millions et trois virgule cinq pour cent.", "Un modèle a 25 millions et 3.5%."),
    ("fr", "quatre vingt dix sept et trois et cinq", "97 et trois et cinq"),
    ("de", "Es sind fünfundzwanzig Prozent.", "Es sind 25%."),
    ("es", "Hay veinticinco por ciento.", "Hay 25%."),
    ("ru", "Это двадцать пять процентов.", "Это 25%."),
    ("th", "มีจำนวนสองล้านคน", "มีจำนวน2000000คน"),
])
def test_native_number_rendering(normalizer, language, source, expected):
    assert normalizer.normalize(source, language) == expected


@pytest.mark.parametrize("language,source", [
    ("en", "I want one of these. Eleven Labs makes speech models."),
    ("zh", "我们一起研究，一直在推进。"),
    ("ja", "今日は新しい音声モデルを紹介します。"),
    ("fr", "Un modèle et une voix."),
    ("th", "จำนวนผู้ใช้เพิ่มขึ้น"),
    ("th", "ความสัมพันธ์และพันธุกรรมมีความสำคัญ"),
    ("th", "มีเชื้อเพลิงหลายล้านแกลลอน"),
    ("th", "ปลาชนิดนี้ศูนย์พันธุ์"),
    ("zh", "每年节省数百万加仑燃料。"),
    ("zh", "一二线城市占百分之四五。"),
    ("en", "The year was 2015, not 2014. The ID is 001234567."),
])
def test_preserve_prose_and_actual_values(normalizer, language, source):
    assert normalizer.normalize(source, language) == source


def test_missing_optional_grammar_preserves_text(tmp_path, caplog):
    renderer = NumericOutputNormalizer(tmp_path)
    source = "GPT-4.1 has twenty voices."
    assert renderer.normalize(source, "en") == source
    assert "preserving text" in caplog.text


def test_unsupported_language_is_unchanged(tmp_path):
    assert NumericOutputNormalizer(tmp_path).normalize("uno due", "it") == "uno due"
