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


def test_photo_summary_preserves_the_adapters_oriented_coordinate_frame():
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
    assert "image_orientation" not in payload
    assert payload["items"][0]["bbox"] == [20, 30, 120, 60]
    assert "画面偏左" in payload["text"]


def test_wide_header_does_not_mix_two_columns_of_wrapped_body():
    from plugins.ocr_summary import _describe_group

    entries = [item("共同主题", [0, 0, 600, 55]),
               item("左栏标题", [0, 80, 120, 100]),
               item("左栏正文第一行", [0, 103, 240, 123]),
               item("左栏正文第二行", [0, 126, 240, 146]),
               item("右栏标题", [350, 80, 470, 100]),
               item("右栏正文第一行", [350, 103, 590, 123]),
               item("右栏正文第二行", [350, 126, 590, 146]),
               item("另一说明", [350, 210, 470, 230])]
    summary = _describe_group(entries)
    assert summary.index("左栏正文第二行") < summary.index("右栏标题")
    assert "组内偏左" in summary and "组内偏右" in summary
    assert "“左栏正文第一行”、“左栏正文第二行”" in summary or "“左栏标题”、“左栏正文第一行”、“左栏正文第二行”" in summary
    assert all(f"“{entry['text']}”" in summary for entry in entries)


def test_lower_caption_is_separated_from_scattered_short_labels():
    from plugins.ocr_summary import _describe_group

    entries = [item("上部", [200, 0, 250, 20]),
               item("左侧一", [0, 60, 60, 80]), item("右侧一", [400, 60, 460, 80]),
               item("中央", [200, 140, 250, 160]),
               item("左侧二", [0, 220, 60, 240]), item("右侧二", [400, 220, 460, 240]),
               item("下方醒目说明", [100, 300, 400, 350])]
    summary = _describe_group(entries)
    assert "组内下方醒目文字为“下方醒目说明”" in summary
    assert "组内中间" in summary and "中部文字块为“中央”" in summary
    assert "环形" not in summary and "连接" not in summary


def test_date_columns_keep_wrapped_lines_and_leave_unrelated_text_separate():
    from plugins.ocr_summary import _describe_group

    entries = [item("无关页眉", [0, 0, 500, 20])]
    for offset, date in enumerate(["2020.1", "2021.2", "2022.3", "2023.4"]):
        x = offset * 150
        entries += [item(date, [x, 100, x + 60, 120]),
                    item(f"第{offset}列原文", [x, 125, x + 90, 145]),
                    item(f"第{offset}列续行", [x, 148, x + 90, 168])]
    summary = _describe_group(entries)
    assert "日期文字按年月顺序整理" in summary
    assert "“2020.1”，同列下方原文为“第0列原文”、“第0列续行”" in summary
    assert summary.index("2020.1") < summary.index("2021.2") < summary.index("2022.3")
    assert "其余文字另列：“无关页眉”" in summary


def test_equally_plausible_date_columns_do_not_force_an_association():
    from plugins.ocr_summary import _dated_blocks

    entries = [item(date, [x, 100, x + 60, 120]) for x, date in
               [(0, "2020"), (40, "2021"), (200, "2022"), (300, "2023")]]
    ambiguous = item("归属不明", [20, 125, 80, 145])
    blocks, remaining = _dated_blocks(entries + [ambiguous])
    assert ambiguous in remaining
    assert all(len(block) == 1 for block in blocks)


def test_repeated_numbers_in_separate_local_blocks_are_not_collapsed():
    from plugins.ocr_summary import _describe_group

    entries = [item("主题说明", [0, 0, 500, 50])]
    for x in [0, 200, 400]:
        entries += [item("4.5", [x, 100, x + 40, 120]),
                    item("4.5", [x, 200, x + 40, 220])]
    assert _describe_group(entries).count("“4.5”") == 6


def test_diagonal_short_list_keeps_the_original_reading_order():
    from plugins.ocr_summary import _describe_group

    entries = [item("第一项", [200, 0, 300, 20]),
               item("第二项", [100, 80, 260, 100]),
               item("第三项", [0, 160, 120, 180])]
    assert _describe_group(entries) == "原文为“第一项”、“第二项”、“第三项”"


def test_numeric_table_keeps_row_order_instead_of_listing_numeric_columns():
    from plugins.ocr_summary import _describe_group

    entries = [item("表格说明", [0, 0, 500, 50])]
    for row in range(3):
        for column in range(3):
            entries.append(item(str(row * 3 + column), [column * 180, 100 + row * 30,
                                                       column * 180 + 40, 120 + row * 30]))
    summary = _describe_group(entries)
    assert "组内" not in summary
    assert summary.index("“0”") < summary.index("“1”") < summary.index("“2”") < summary.index("“3”")


def test_date_column_continuation_can_extend_beyond_the_first_line_window():
    from plugins.ocr_summary import _dated_blocks

    entries = [item(date, [x, 100, x + 60, 120]) for x, date in
               [(0, "2020"), (150, "2021"), (300, "2022"), (450, "2023")]]
    lines = [item(f"续行{i}", [450, 125 + i * 24, 540, 145 + i * 24]) for i in range(9)]
    blocks, remaining = _dated_blocks(entries + list(reversed(lines)))
    assert blocks[-1][1:] == lines
    assert remaining == []


def test_overlapping_slanted_boxes_do_not_turn_one_column_into_parallel_columns():
    from plugins.ocr_summary import _describe_group

    entries = [item("上方醒目主题", [100, 0, 700, 100]),
               item("第一段完整原文", [100, 150, 450, 200]),
               item("第二段较长的原文", [200, 300, 650, 440]),
               item("第二段的续行", [200, 340, 450, 420]),
               item("第三段完整原文", [200, 500, 650, 560])]
    summary = _describe_group(entries)
    assert "组内" not in summary
    assert "候选标题" in summary
