"""Read-only spatial captions and MOSS prompt compilation.

MOSS layout compilation follows OpenMOSS/MOSS-Image, Apache-2.0,
1961ca3d111b828e65278656cde0c8f94484f4dd:
skills/moss-image-prompt-enhancer/scripts/compile_prompt.py.
GBC input follows KohakuBlueleaf/KohakuGBC, Apache-2.0,
d90f360dbf561f63270522e44fa0e74ebed59c12 (resolved graph JSON).
GBC conversion follows prompt/gbc.py and prompt/moss.py with relations=True.
https://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpatialCaption:
    format: str
    document: dict

    def text(self) -> str:
        if self.format == "moss_layout":
            return self.document["summary"]["caption"]
        root = next(v for v in _graph(self.document)["vertices"] if v["label"] == "image")
        return _preferred_caption(root) or ""


def detect_spatial_format(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    if "vertices" in data or isinstance(data.get("graph"), dict) and "vertices" in data["graph"]:
        return "gbc"
    if "items" in data and "summary" in data or "bbox_coord_space" in data or "bbox_format" in data:
        return "moss_layout"
    return None


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _fields(value: Any, allowed: set[str], required: set[str], where: str) -> None:
    _require(isinstance(value, dict), f"{where} 必须是对象")
    _require(not (value.keys() - allowed), f"{where} 含不支持的字段：{sorted(value.keys() - allowed)}")
    _require(not (required - value.keys()), f"{where} 缺少字段：{sorted(required - value.keys())}")


def _text(value: Any, where: str, *, empty: bool = False) -> None:
    _require(isinstance(value, str) and (empty or bool(value.strip())), f"{where} 必须是{'文本' if empty else '非空文本'}")


def _number(value: Any, where: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{where} 必须是有限数值")
    return float(value)


def _strings(value: Any, where: str) -> None:
    _require(isinstance(value, list), f"{where} 必须是文本数组")
    for i, item in enumerate(value):
        _text(item, f"{where}[{i}]")


def _box(value: Any, where: str) -> tuple[float, float, float, float]:
    _require(isinstance(value, list) and len(value) == 4, f"{where} 必须含四个坐标")
    a, b, c, d = (_number(v, where) for v in value)
    _require(a < c and b < d, f"{where} 必须具有正宽度和高度")
    return a, b, c, d


def _layout(data: dict) -> None:
    canonical = data.get("bbox_coord_space") == "abs_centralized"
    keys = {"summary", "items", "bbox_coord_space", "bbox_format"}
    if not canonical:
        keys |= {"width", "height"}
    _fields(data, keys, keys, "Layout")
    _require(data["bbox_coord_space"] == ("abs_centralized" if canonical else "abs"), "Layout 坐标空间必须是 abs 或 abs_centralized")
    _require(data["bbox_format"] == ("yxyx" if canonical else "xyxy"), "Layout 坐标顺序与坐标空间不匹配")
    if not canonical:
        for key in ("width", "height"):
            v = data[key]
            _require(type(v) is int and v > 0 and v % 16 == 0, f"Layout {key} 必须是 16 的正整数倍")
    summary = data["summary"]
    _fields(summary, {"caption", "meta", "canvas", "palette"}, {"caption"}, "summary")
    _text(summary["caption"], "summary.caption")
    if "canvas" in summary:
        _require(summary["canvas"] in ("single", "multi_panel", "collage"), "summary.canvas 不支持此值")
    for key in ("meta", "palette"):
        if key in summary:
            _strings(summary[key], f"summary.{key}")
    _require(isinstance(data["items"], list) and bool(data["items"]), "Layout items 必须是非空数组")
    for i, item in enumerate(data["items"]):
        where = f"items[{i}]"
        _fields(item, {"label", "caption", "bbox", "text", "tags", "palette"}, {"label", "caption", "bbox"}, where)
        _require(item["label"] in ("object", "character", "text"), f"{where}.label 不支持此值")
        _text(item["caption"], f"{where}.caption")
        if item["label"] == "text":
            _require("text" in item, f"{where} 缺少实际文字 text")
        if "text" in item:
            _text(item["text"], f"{where}.text")
        for key in ("tags", "palette"):
            if key in item:
                _strings(item[key], f"{where}.{key}")
        a, b, c, d = _box(item["bbox"], f"{where}.bbox")
        if canonical:
            _require(all(type(v) is int for v in item["bbox"]), f"{where}.bbox 的中心坐标必须是整数")
        else:
            _require(0 <= a < c <= data["width"] and 0 <= b < d <= data["height"], f"{where}.bbox 超出 Layout 画布")


def _graph(data: dict) -> dict:
    return data["graph"] if "graph" in data and "vertices" not in data else data


def _desc(desc: Any, where: str) -> tuple[str, str]:
    if isinstance(desc, dict):
        label, text = desc.get("label"), desc.get("text")
    else:
        _require(isinstance(desc, list) and len(desc) == 2, f"{where} 必须含 label 和 text")
        label, text = desc
    _text(label, f"{where}.label")
    _text(text, f"{where}.text")
    return label, text


def _scores(value: Any, where: str) -> None:
    _require(isinstance(value, dict), f"{where} 必须是名称到置信度的对象")
    for tag, score in value.items():
        _text(tag, where)
        _require(0 <= _number(score, where) <= 1, f"{where} 置信度必须在 0 到 1 之间")


def _gbc(data: dict) -> None:
    graph = _graph(data)
    _require(isinstance(graph, dict), "GBC graph 必须是对象")
    vs = graph.get("vertices")
    _require(isinstance(vs, list) and bool(vs), "GBC vertices 必须是非空数组")
    by_id = {}
    for i, v in enumerate(vs):
        _require(isinstance(v, dict), f"vertices[{i}] 必须是对象")
        _text(v.get("vertex_id"), f"vertices[{i}].vertex_id", empty=True)
        vid = v["vertex_id"]
        _require(vid not in by_id, f"GBC vertex_id 重复：{vid}")
        by_id[vid] = v
        _require(v.get("label") in ("image", "entity", "composition", "relation"), f"vertices[{i}].label 不支持此值")
        _require(isinstance(v.get("descs"), list), f"vertices[{i}].descs 必须是数组")
        for j, d in enumerate(v["descs"]):
            _desc(d, f"vertices[{i}].descs[{j}]")
        if v.get("bbox") is not None:
            b = v["bbox"]
            _fields(b, {"left", "top", "right", "bottom", "confidence"}, {"left", "top", "right", "bottom"}, f"vertices[{i}].bbox")
            box = _box([b[k] for k in ("left", "top", "right", "bottom")], f"vertices[{i}].bbox")
            _require(all(0 <= x <= 1 for x in box), "GBC bbox 坐标必须在 0 到 1 之间")
            if b.get("confidence") is not None:
                _require(0 <= _number(b["confidence"], "bbox.confidence") <= 1, "bbox.confidence 必须在 0 到 1 之间")
        tags = v.get("tags")
        if tags is not None:
            _scores(tags, "GBC vertex tags")
    roots = [v["vertex_id"] for v in vs if v["label"] == "image"]
    _require(len(roots) == 1, "GBC 必须且只能有一个 image 节点")
    edges: set[tuple[str, str, str]] = set()
    for v in vs:
        _require(isinstance(v.get("out_edges"), list), "GBC out_edges 必须是数组")
        for edge in v["out_edges"]:
            _fields(edge, {"source", "target", "text"}, {"source", "target", "text"}, "GBC edge")
            for key in ("source", "target", "text"):
                _text(edge[key], f"edge.{key}", empty=True)
            _require(edge["source"] == v["vertex_id"] and edge["target"] in by_id, "GBC edge 引用了缺失或不匹配的节点")
            key = (edge["source"], edge["target"], edge["text"])
            _require(key not in edges, "GBC edge 重复")
            edges.add(key)
        for key in ("sub_masks", "super_masks"):
            if key in v:
                _require(isinstance(v[key], list), f"GBC {key} 必须是数组")
                _require(all(isinstance(x, str) and x in by_id for x in v[key]), f"GBC {key} 引用了缺失节点")
    for v in vs:
        if "in_edges" in v:
            _require(isinstance(v["in_edges"], list), "GBC in_edges 必须是数组")
            incoming = []
            for edge in v["in_edges"]:
                _fields(edge, {"source", "target", "text"}, {"source", "target", "text"}, "GBC in_edge")
                for key in ("source", "target", "text"):
                    _text(edge[key], f"in_edge.{key}", empty=True)
                incoming.append((edge["source"], edge["target"], edge["text"]))
            expected = {e for e in edges if e[1] == v["vertex_id"]}
            _require(len(incoming) == len(set(incoming)) and set(incoming) == expected, "GBC in_edges 与 out_edges 不一致")
    # Iterative traversal preserves shared descendants without recursing through captions.
    visiting, seen = set(), set()
    stack = [(roots[0], False)]
    while stack:
        vid, done = stack.pop()
        if done:
            visiting.remove(vid)
            seen.add(vid)
        elif vid not in seen:
            _require(vid not in visiting, "GBC 图包含循环")
            visiting.add(vid)
            stack.append((vid, True))
            stack.extend((e["target"], False) for e in reversed(by_id[vid]["out_edges"]))
    _require(len(seen) == len(vs), "GBC 存在无法从 image 节点到达的节点")
    if graph.get("img_size") is not None:
        size = graph["img_size"]
        _require(isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size), "GBC img_size 必须是两个正整数")
    for key in ("original_caption", "short_caption", "detail_caption", "img_path", "img_url"):
        if graph.get(key) is not None:
            _text(graph[key], f"GBC {key}", empty=True)
    if "mask_inside_threshold" in graph:
        _require(0 <= _number(graph["mask_inside_threshold"], "mask_inside_threshold") <= 1, "mask_inside_threshold 必须在 0 到 1 之间")
    if "tags" in graph:
        tags = graph["tags"]
        _fields(tags, {"image", "categories", "regions", "crops"}, {"image", "categories", "regions", "crops"}, "GBC graph tags")
        _scores(tags["image"], "GBC image tags")
        _require(isinstance(tags["categories"], dict), "GBC categories 必须是对象")
        for tag, category in tags["categories"].items():
            _text(tag, "GBC category tag")
            _text(category, "GBC category")
        for key in ("regions", "crops"):
            _require(isinstance(tags[key], dict), f"GBC {key} 必须是对象")
            for name, scores in tags[key].items():
                _text(name, f"GBC {key} name", empty=True)
                _scores(scores, f"GBC {key} scores")


def parse_spatial_caption(data: Any, *, filename: str = "caption.json") -> SpatialCaption:
    try:
        fmt = detect_spatial_format(data)
        _require(fmt is not None, "不是 GBC 或 MOSS Layout 标注")
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        (_layout if fmt == "moss_layout" else _gbc)(data)
        return SpatialCaption(fmt, copy.deepcopy(data))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{filename}: {exc}") from exc


def _geometry(geometry: dict | None) -> dict | None:
    if geometry is None:
        return None
    for key in ("source_width", "source_height", "width", "height", "resized_width", "resized_height"):
        _require(type(geometry.get(key)) is int and geometry[key] > 0, f"图像几何 {key} 必须是正整数")
    for key in ("left", "top"):
        _require(type(geometry.get(key)) is int and geometry[key] >= 0, f"图像几何 {key} 必须是非负整数")
    return geometry


def _mapped_box(box, width, height, geometry, image_fit, flip):
    left, top, right, bottom = box
    target_w, target_h = width, height
    if geometry is not None:
        g = geometry
        _require(width * g["source_height"] == height * g["source_width"], "Layout 画布比例与原图不一致")
        sx, sy = g["resized_width"] / width, g["resized_height"] / height
        sign = 1 if image_fit == "pad" else -1
        left, right = left * sx + sign * g["left"], right * sx + sign * g["left"]
        top, bottom = top * sy + sign * g["top"], bottom * sy + sign * g["top"]
        target_w, target_h = g["width"], g["height"]
    clipped = [max(0.0, left), max(0.0, top), min(float(target_w), right), min(float(target_h), bottom)]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None, True, target_w, target_h
    partial = clipped != [left, top, right, bottom]
    if flip:
        clipped[0], clipped[2] = target_w - clipped[2], target_w - clipped[0]
    return clipped, partial, target_w, target_h


def _layout_items(data, geometry, image_fit, flip):
    canonical = data["bbox_coord_space"] == "abs_centralized"
    if canonical and geometry is None:
        _require(not flip, "翻转 MOSS Layout 需要原图尺寸")
        return copy.deepcopy(data["items"]), None, None
    width, height = (geometry["source_width"], geometry["source_height"]) if canonical else (data["width"], data["height"])
    result = []
    target_w, target_h = width, height
    for item in data["items"]:
        box = item["bbox"]
        if canonical:
            y1, x1, y2, x2 = box
            box = [x1 + width // 2, y1 + height // 2, x2 + width // 2, y2 + height // 2]
            _require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, "MOSS Layout 中心坐标超出原图画布")
        box, partial, target_w, target_h = _mapped_box(box, width, height, geometry, image_fit, flip)
        if box is None:
            continue
        if item["label"] == "text" or "text" in item:
            _require(not partial, "裁剪会截断 Layout 文字框，请调整裁剪位置或改用填充")
            _require(not flip, "含文字的 Layout 不支持水平翻转，请关闭数据集翻转")
        new = copy.deepcopy(item)
        x1, y1, x2, y2 = box
        # Same edge enclosure and centre convention as the official compiler.
        new["bbox"] = [math.floor(round(y1, 9)) - target_h // 2, math.floor(round(x1, 9)) - target_w // 2,
                       math.ceil(round(y2, 9)) - target_h // 2, math.ceil(round(x2, 9)) - target_w // 2]
        _require(new["bbox"][0] < new["bbox"][2] and new["bbox"][1] < new["bbox"][3], "Layout 框小于坐标导出精度")
        result.append(new)
    _require(bool(result), "裁剪后没有可见的 Layout 元素，请调整裁剪位置或改用填充")
    return result, target_w, target_h


# Defaults of the pinned from_gbc renderer. Relations are explicitly enabled below.
_GBC_CAPTIONS = ("detail", "composition", "short", "hardcode")
_GBC_TAG_THRESHOLD = 0.35
_SPACE = re.compile(r"\s+")
_BOX_LIKE = re.compile(r"\[(-?[0-9]+),(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)\]( |$)")


def _preferred_caption(vertex: dict) -> str | None:
    descriptions = dict(reversed([_desc(d, "GBC desc") for d in vertex["descs"]]))
    return next((descriptions[key] for key in _GBC_CAPTIONS if key in descriptions), None)


def _clean(value: str | None) -> str | None:
    return (_SPACE.sub(" ", value).strip() or None) if value is not None else None


def _gbc_draft(data: dict, geometry: dict | None, image_fit: str, flip: bool) -> dict:
    """Pinned from_gbc -> to_moss semantics, retaining relations of visible participants."""
    graph = _graph(data)
    declared = graph.get("img_size")
    if geometry is None:
        _require(declared is not None, "GBC 转 MOSS 需要原图尺寸，请提供图片几何或 img_size")
        width, height = declared
        target_w, target_h = width, height
    else:
        width, height = geometry["source_width"], geometry["source_height"]
        target_w, target_h = geometry["width"], geometry["height"]
        if declared is not None:
            _require(declared[0] * height == declared[1] * width, "GBC img_size 比例与原图不一致")
    _require(target_w % 16 == 0 and target_h % 16 == 0, "GBC 转 MOSS 的目标画布必须是 16 的整数倍")
    by_id = {v["vertex_id"]: v for v in graph["vertices"]}
    root = next(v["vertex_id"] for v in graph["vertices"] if v["label"] == "image")

    def kids(vid, relation):
        edges = {e["target"]: e for e in by_id[vid]["out_edges"]}
        return [e for target, e in edges.items() if (by_id[target]["label"] == "relation") == relation]

    # The first depth-first parent assigns a handle and owns each shared descendant.
    handles, used = {root: "image"}, {"image"}
    order = []
    stack = list(reversed(kids(root, False)))
    while stack:
        edge = stack.pop()
        vid = edge["target"]
        if vid in handles:
            continue
        base = (_clean(edge["text"]) or _clean(vid) or "vertex").replace("[", "(").replace("]", ")")
        name, index = base, 1
        while name in used:
            index += 1
            name = f"{base} {index}"
        _require(_BOX_LIKE.match(f"[{name}]") is None, "GBC 节点名称不能与坐标框格式混淆")
        handles[vid] = name
        used.add(name)
        order.append(vid)
        stack.extend(reversed(kids(vid, False)))
    _require(all(v["label"] == "relation" or vid in handles for vid, v in by_id.items()), "GBC 节点缺少非关系父节点路径")
    captions = {vid: _clean(_preferred_caption(by_id[vid])) for vid in handles}
    for vid, caption in captions.items():
        _require(vid != root and by_id[vid].get("bbox") is not None or caption is None or _BOX_LIKE.match(caption) is None,
                 "GBC 无框节点描述不能以坐标框格式开头")
    converted_tags = {}
    for vid in handles:
        vertex = by_id[vid]
        tags = [tag.replace("_", " ") for tag, score in
                sorted((vertex.get("tags") or {}).items(), key=lambda pair: -pair[1])
                if score >= _GBC_TAG_THRESHOLD]
        _require(all(tag and tag == tag.strip() and "\n" not in tag and "," not in tag for tag in tags),
                 "GBC 转 MOSS 的标签不能含逗号、换行或首尾空格")
        converted_tags[vid] = tags

    visible = {root}
    items = []
    for vid in order:
        vertex = by_id[vid]
        tags = converted_tags[vid]
        bbox = vertex.get("bbox")
        if bbox is None:
            visible.add(vid)
            continue
        box = [bbox["left"] * width, bbox["top"] * height, bbox["right"] * width, bbox["bottom"] * height]
        mapped, _, _, _ = _mapped_box(box, width, height, geometry, image_fit, flip)
        if mapped is None:
            continue
        visible.add(vid)
        item = {"label": "object", "caption": captions[vid] or handles[vid]}
        if tags:
            item["tags"] = tags
        item["bbox"] = box
        items.append(item)

    relations = []
    # to_moss visits the root and then first-host DFS order, collecting each node's relations.
    for vid in [root, *order]:
        for edge in kids(vid, True):
            relation = by_id[edge["target"]]
            participants = list(dict.fromkeys(e["target"] for e in relation["out_edges"]))
            _require(all(p in handles for p in participants), "GBC 关系参与者缺失或不是实体节点")
            for desc in relation["descs"]:
                label, text = _desc(desc, "GBC relation")
                text = _clean(text)
                if label == "relation" and text:
                    _require(bool(participants), "GBC 关系缺少参与者")
                    if vid in visible and all(p in visible for p in participants):
                        relations.append(text.replace("[", "(").replace("]", ")"))
    caption = captions[root] or "image"
    if relations:
        caption = " ".join([caption, *relations])
    return {"width": width, "height": height, "bbox_coord_space": "abs", "bbox_format": "xyxy",
            "summary": {"canvas": "single", "caption": caption}, "items": items}


def render_spatial_caption(
    raw: SpatialCaption, *, mode: str, geometry: dict | None = None,
    image_fit: str = "crop", flip: bool = False,
) -> str:
    _require(mode in ("native", "gbc_to_moss"), "不支持此结构标注转换方式")
    _require(image_fit in ("crop", "pad"), "不支持此图像适配方式")
    geometry = _geometry(geometry)
    if raw.format == "gbc":
        _require(mode == "gbc_to_moss", "GBC 标注需要选择转为 MOSS Layout")
        document = _gbc_draft(raw.document, geometry, image_fit, flip)
    else:
        _require(raw.format == "moss_layout", "不支持此结构标注格式")
        _require(mode == "native", "MOSS Layout 标注应使用原生模式")
        document = raw.document
    items, _, _ = _layout_items(document, geometry, image_fit, flip)
    result = {"summary": document["summary"], "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
