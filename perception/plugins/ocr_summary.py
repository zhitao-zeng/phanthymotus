"""Describe OCR text layout with fixed templates and no image/model calls.

Regions are text groups inferred from whitespace, not detected screens or
boards. Single fragments are valid groups, so a missing physical outline
never prevents a summary. The raw OCR payload remains the evidence record.
"""
from __future__ import annotations

from statistics import median
import re
import unicodedata


_NUMBER_TEXT = re.compile(
    r"[¥$€£]?[+-]?[0-9OoIlLSsBZbz][0-9OoIlLSsBZbz,.]*(?:[x×]\d+)?"
    r"(?:万|千|亿)?(?:%|\+|元|人|家|项|次|台|个|年|月|天|米|秒|kg|km|cm|mm|m|g)?[↑↓]?"
)


def _reading_order(item):
    x1, y1, x2, y2 = item["bbox"]
    return y1, x1, y2, x2, item["text"]


def _gaps(items, axis):
    intervals = sorted((item["bbox"][axis], item["bbox"][axis + 2]) for item in items)
    end = intervals[0][1]
    gaps = []
    for start, next_end in intervals[1:]:
        if start > end:
            gaps.append((start - end, (start + end) / 2))
        end = max(end, next_end)
    return gaps


def _layout_groups(items):
    if len(items) < 2:
        return [items]
    height = median(max(1, item["bbox"][3] - item["bbox"][1]) for item in items)
    splits = []
    # A larger vertical allowance keeps ordinary line spacing together.
    for axis, factor in ((0, 2.5), (1, 4.0)):
        for gap, split in _gaps(items, axis):
            if gap >= factor * height:
                splits.append((gap / (factor * height), -axis, split))
    if not splits:
        return [items]
    _, neg_axis, split = max(splits)
    axis = -neg_axis
    first = [item for item in items if item["bbox"][axis + 2] <= split]
    second = [item for item in items if item["bbox"][axis] >= split]
    return _layout_groups(first) + _layout_groups(second)


def _font_size(item):
    """Conservative glyph-size proxy; a tilted long line has a tall bbox."""
    x1, y1, x2, y2 = item["bbox"]
    units = sum(1.0 if unicodedata.east_asian_width(char) in {"W", "F"} else .5
                for char in item["text"] if not char.isspace())
    return min(max(1, y2 - y1), max(1, x2 - x1) / max(1, units))


def _unique_items(items):
    # Repeated numeric observations may be different table cells or prices.
    # Only literal non-numeric repeats can share a presentation entry.
    unique = {}
    numeric = []
    for item in sorted(items, key=_reading_order):
        if any(char.isdigit() for char in item["text"]):
            numeric.append(item)
            continue
        previous = unique.get(item["text"])
        if previous is None or _font_size(item) > _font_size(previous):
            unique[item["text"]] = item
    return sorted([*unique.values(), *numeric], key=_reading_order)


def _quotes(items):
    return "、".join(
        f"“{item['text']}”" + ("（OCR 分数较低，待核实）" if item.get("score", 1.) < .95 else "")
        for item in _unique_items(items)
    )


def _title_candidates(members):
    if len(members) < 2:
        return []
    top = min(item["bbox"][1] for item in members)
    bottom = max(item["bbox"][3] for item in members)
    upper = []
    for item in members:
        text = unicodedata.normalize("NFKC", item["text"]).replace(" ", "")
        numeric = any(char.isdigit() for char in text) and _NUMBER_TEXT.fullmatch(text)
        if not numeric and item["bbox"][1] <= top + .4 * (bottom - top):
            upper.append(item)
    if not upper:
        return []
    largest = max(_font_size(item) for item in upper)
    typical = median(_font_size(item) for item in members)
    if largest < 1.35 * typical:
        return []
    return [item for item in upper if _font_size(item) >= .8 * largest]


def _describe_group(members):
    members = _unique_items(members)
    titles = _title_candidates(members)
    if not titles:
        return f"原文为{_quotes(members)}"
    first = min(sum(item["bbox"][1::2]) / 2 for item in titles)
    last = max(sum(item["bbox"][1::2]) / 2 for item in titles)
    other = [item for item in members if not any(item is title for title in titles)]
    before = [item for item in other if sum(item["bbox"][1::2]) / 2 < first]
    after = [item for item in other if sum(item["bbox"][1::2]) / 2 > last]
    nearby = [item for item in other if item not in before and item not in after]
    clauses = [f"候选标题为{_quotes(titles)}"]
    for label, entries in [("上方另有", before), ("下方原文为", after), ("附近另有", nearby)]:
        if entries:
            clauses.append(f"{label}{_quotes(entries)}")
    return "，".join(clauses)


def _describe_zone(regions):
    # Identical text in distinct locations is reported as repetition, not as
    # independent exhibits or an inferred shared source.
    repeated = {}
    for _position, _y, _x, members in regions:
        key = tuple(item["text"] for item in _unique_items(members))
        repeated.setdefault(key, []).append(members)
    clauses = []
    for occurrences in repeated.values():
        contains_numbers = any(char.isdigit() for item in occurrences[0] for char in item["text"])
        if len(occurrences) > 1 and not contains_numbers:
            representative = [item for members in occurrences for item in members]
            clauses.append(f"{len(occurrences)}处重复出现{_quotes(representative)}")
        else:
            clauses.extend(_describe_group(members) for members in occurrences)
    prefix = "按从上到下的顺序，" if len(clauses) > 1 else ""
    return prefix + "；另一个文字组中，".join(clauses)


def _summary_coordinates(items, image_size, orientation):
    if orientation == 1:
        return items, image_size
    width, height = image_size
    transforms = {
        2: lambda x, y: (width - x, y),
        3: lambda x, y: (width - x, height - y),
        4: lambda x, y: (x, height - y),
        5: lambda x, y: (y, x),
        6: lambda x, y: (height - y, x),
        7: lambda x, y: (height - y, width - x),
        8: lambda x, y: (y, width - x),
    }
    transform = transforms[orientation]
    displayed = []
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        corners = [transform(x, y) for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]]
        xs, ys = zip(*corners)
        displayed.append({**item, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return displayed, (height, width) if orientation >= 5 else image_size


def build_ocr_summary(items, image_size=None, image_orientation=1):
    """Return one paragraph, including every non-empty OCR fragment.

    Without frame dimensions, text is still returned but no left/right
    position is invented. Whitespace grouping always keeps singleton text.
    Numbers and labels are independent quoted observations. Only the summary
    uses display-oriented coordinates; input items are never modified.
    """
    readable = sorted((item for item in items if item.get("text")), key=_reading_order)
    if not readable:
        return "当前画面没有识别到可用文字。"
    if image_size is None:
        return f"识别到的文字内容是{_quotes(readable)}。"

    readable, image_size = _summary_coordinates(readable, image_size, image_orientation)

    width, height = image_size
    regions = []
    for members in _layout_groups(readable):
        members = sorted(members, key=_reading_order)
        x1 = min(item["bbox"][0] for item in members)
        y1 = min(item["bbox"][1] for item in members)
        x2 = max(item["bbox"][2] for item in members)
        y2 = max(item["bbox"][3] for item in members)
        x = (x1 + x2) / (2 * width)
        y = (y1 + y2) / (2 * height)
        position = "偏左" if x < 1 / 3 else "偏右" if x > 2 / 3 else "中间"
        regions.append((position, y, x, members))
    # Center is described first as a spatial convention, not a user-intent vote.
    regions.sort(key=lambda region: (
        {"中间": 0, "偏左": 1, "偏右": 2}[region[0]], region[1], region[2],
        tuple(_reading_order(item) for item in region[3]),
    ))
    descriptions = []
    for position in ["中间", "偏左", "偏右"]:
        located = [region for region in regions if region[0] == position]
        if located:
            descriptions.append(f"画面{position}：{_describe_zone(located)}")
    return "。".join(descriptions) + "。"
