"""Text-layout summaries must work without detected physical regions."""
import copy
from io import BytesIO

import pytest

from plugins.ocr_runtime import build_ocr_payload, recognize_to_payload
from plugins.ocr_summary import build_ocr_summary, _summary_coordinates


def item(text, box):
    return {"text": text, "bbox": box, "score": .99}


def test_coordinate_only_summary_keeps_each_side_separate():
    items = [item("左侧标题", [20, 40, 120, 60]),
             item("左侧说明", [20, 75, 120, 95]),
             item("中央标题", [220, 40, 320, 60]),
             item("中央说明", [220, 75, 320, 95]),
             item("右侧标题", [420, 40, 520, 60])]
    summary = build_ocr_summary(items, (600, 300))
    assert "画面中间：原文为“中央标题”、“中央说明”" in summary
    assert "画面偏左：原文为“左侧标题”、“左侧说明”" in summary
    assert "画面偏右：原文为“右侧标题”" in summary


def test_singleton_never_needs_a_boundary_or_another_text():
    summary = build_ocr_summary([item("孤立文字", [500, 80, 560, 100])], (600, 300))
    assert "偏右" in summary and "“孤立文字”" in summary


def test_no_frame_dimensions_still_returns_all_text_without_guessing_position():
    summary = build_ocr_summary([item("单独文字", [10, 10, 100, 30])])
    assert summary == "识别到的文字内容是“单独文字”。"


def test_stacked_groups_and_fragments_are_not_lost():
    items = [item("上部文字", [200, 20, 300, 40]),
             item("下部文字", [200, 220, 300, 240]),
             item("7", [540, 180, 550, 200])]
    summary = build_ocr_summary(items, (600, 300))
    assert all(f"“{entry['text']}”" in summary for entry in items)
    assert summary.index("上部文字") < summary.index("下部文字")


def test_wrong_words_and_numbers_are_not_corrected_or_associated():
    items = [item("S0万+", [20, 20, 100, 40]),
             item("50+", [20, 55, 100, 75]),
             item("合作伙伴", [20, 90, 100, 110]),
             item("开R合作", [20, 125, 100, 145])]
    summary = build_ocr_summary(items, (600, 300))
    assert all(f"“{entry['text']}”" in summary for entry in items)
    assert "50家" not in summary and "50万+" not in summary and "开放合作" not in summary


def test_literal_duplicates_in_different_groups_stay_separate():
    items = [item("相同名称", [20, 20, 100, 40]),
             item("相同名称", [20, 55, 100, 75]),
             item("相同名称", [450, 20, 550, 40])]
    assert build_ocr_summary(items, (600, 300)).count("“相同名称”") == 2


def test_summary_is_order_independent_and_does_not_modify_items():
    items = [item("右侧", [400, 20, 500, 40]), item("左侧", [20, 20, 100, 40])]
    before = copy.deepcopy(items)
    assert build_ocr_summary(items, (600, 300)) == build_ocr_summary(list(reversed(items)), (600, 300))
    assert items == before


def test_regions_are_never_claimed_to_be_detected_physical_objects():
    summary = build_ocr_summary([item("文字", [20, 20, 100, 40])], (600, 300))
    assert all(word not in summary for word in ["屏幕", "展板", "笔记本", "主来源"])


def test_repeated_text_is_explained_instead_of_numbered_as_two_exhibits():
    items = [item("重复名称", [220, 20, 320, 40]), item("重复名称", [220, 220, 320, 240])]
    summary = build_ocr_summary(items, (600, 300))
    assert "2处重复出现“重复名称”" in summary
    assert "区域1" not in summary and "区域2" not in summary


def test_title_and_spatial_body_are_explicit_candidates():
    items = [item("说明性页眉", [50, 10, 170, 30]),
             item("醒目主题", [50, 40, 210, 80]),
             item("较小的第一段正文", [50, 85, 210, 105]),
             item("较小的第二段正文", [50, 110, 210, 130])]
    summary = build_ocr_summary(items, (600, 300))
    assert "候选标题为“醒目主题”" in summary
    assert "上方另有“说明性页眉”" in summary
    assert "下方原文为“较小的第一段正文”、“较小的第二段正文”" in summary


def test_tilted_long_body_is_not_selected_just_because_bbox_is_tall():
    items = [item("主题", [50, 20, 150, 70]),
             item("这是一段很长的正文文字需要原样保留", [50, 75, 250, 170]),
             item("另一段正文继续说明其他内容", [50, 175, 250, 195])]
    summary = build_ocr_summary(items, (600, 300))
    assert "候选标题为“主题”" in summary
    assert "候选标题为“这是一段" not in summary
    assert all(f"“{entry['text']}”" in summary for entry in items)


def test_low_score_fragments_are_preserved_and_marked_for_verification():
    entry = item("识别残片原文", [20, 20, 100, 40])
    entry["score"] = .92
    assert "“识别残片原文”（OCR 分数较低，待核实）" in build_ocr_summary([entry], (600, 300))


def test_bare_numbers_do_not_become_titles_because_they_are_large():
    for value in ["95%", "50+", "S0万+", "12人", "129.2%↑", "30%↓", "￥20", "$35"]:
        entries = [item(value, [20, 20, 220, 100]),
                   item("普通原文说明", [20, 105, 220, 125]),
                   item("另一条原文说明", [20, 130, 220, 150])]
        summary = build_ocr_summary(entries, (600, 300))
        assert f"“{value}”" in summary
        assert f"候选标题为“{value}”" not in summary


def test_repeated_numeric_cells_are_preserved_inside_one_group():
    entries = [item("甲项", [20, 20, 100, 40]),
               item("4.5", [130, 20, 180, 40]),
               item("乙项", [20, 55, 100, 75]),
               item("4.5", [130, 55, 180, 75]),
               item("4.5", [200, 55, 250, 75])]
    summary = build_ocr_summary(entries, (600, 300))
    assert summary.count("“4.5”") == 3


def test_repeated_prices_are_not_collapsed_as_a_repeated_title():
    entries = [item("2元", [220, 20, 280, 40]), item("2元", [220, 220, 280, 240])]
    summary = build_ocr_summary(entries, (600, 300))
    assert summary.count("“2元”") == 2
    assert "重复出现" not in summary


def test_numeric_named_title_does_not_remove_a_later_identical_observation():
    entries = [item("主题2026", [20, 20, 220, 70]),
               item("这是较小的正文说明", [20, 75, 220, 95]),
               item("主题2026", [20, 100, 100, 120]),
               item("另一条较小的正文说明", [20, 125, 220, 145])]
    summary = build_ocr_summary(entries, (600, 300))
    assert "候选标题为“主题2026”" in summary
    assert summary.count("“主题2026”") == 2


def test_named_technology_with_a_digit_can_still_be_a_title_candidate():
    entries = [item("5G基站", [20, 20, 220, 70]),
               item("这是正文原文说明", [20, 75, 220, 95]),
               item("另一段正文原文说明", [20, 100, 220, 120])]
    assert "候选标题为“5G基站”" in build_ocr_summary(entries, (600, 300))


def test_payload_text_describes_layout_and_keeps_raw_ocr_coordinates():
    items = [item("左栏", [20, 20, 100, 40]), item("右栏", [450, 20, 550, 40])]
    payload = build_ocr_payload(items, 1.0, "zh", image_size=(600, 300))
    assert "画面" in payload["text"]
    assert [entry["text"] for entry in payload["items"]] == ["左栏", "右栏"]
    assert "summary" not in payload
    assert "“左栏”" in payload["text"] and "“右栏”" in payload["text"]
    assert [entry["bbox"] for entry in payload["items"]] == [entry["bbox"] for entry in items]
    assert payload["image_size"] == [600, 300]


def test_empty_and_error_payloads_are_described_without_invented_content():
    assert build_ocr_payload([], 0., "zh")["text"] == "当前画面没有识别到可用文字。"
    failed = build_ocr_payload([], 0., "zh", error=RuntimeError("failed"))
    assert failed["error"] == "failed"
    assert failed["text"] == "文字识别失败，暂时无法生成描述。"


@pytest.mark.parametrize("orientation,box,size", [
    (1, [20, 30, 120, 60], (600, 300)),
    (2, [480, 30, 580, 60], (600, 300)),
    (3, [480, 240, 580, 270], (600, 300)),
    (4, [20, 240, 120, 270], (600, 300)),
    (5, [30, 20, 60, 120], (300, 600)),
    (6, [240, 20, 270, 120], (300, 600)),
    (7, [240, 480, 270, 580], (300, 600)),
    (8, [30, 480, 60, 580], (300, 600)),
])
def test_summary_coordinate_transform_keeps_raw_observation(orientation, box, size):
    entries = [item("方向测试", [20, 30, 120, 60])]
    before = copy.deepcopy(entries)
    transformed, display_size = _summary_coordinates(entries, (600, 300), orientation)
    assert transformed[0]["bbox"] == box
    assert display_size == size
    assert entries == before


def test_real_photo_exif_changes_summary_direction_without_changing_box_reference():
    from PIL import Image

    original = [item("方向测试", [20, 30, 120, 60])]
    class Adapter:
        def recognize(self, image_bytes, language):
            return original

    image = Image.new("RGB", (600, 300), "white")
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    payload = recognize_to_payload(Adapter(), buffer.getvalue(), "zh", 0.)
    assert payload["image_size"] == [600, 300]
    assert payload["image_orientation"] == 6
    assert payload["items"][0]["bbox"] == [20, 30, 120, 60]
    assert "画面偏右" in payload["text"]
