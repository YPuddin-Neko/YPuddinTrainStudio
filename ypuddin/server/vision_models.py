"""Local ONNX vision models: image taggers and the anime head detector.

Inference runs in a spawned child process so the service never loads ONNX Runtime itself and every
job starts from a clean runtime (a CUDA provider that fails falls back to the CPU for the rest of
the job). Only the ONNX graph, its external weights and its label table are read; no model
repository code is executed.

Taggers (``model_catalog.VISION_MODELS``, role ``tagger``) differ in how they read an image and name
their labels; the catalog entry says which (``preprocess``, ``input_size``, ``labels``, ``output``):

- ``wd``: WD v3 / MOAT v2. White square, LANCZOS, BGR 0..255, NHWC. Probabilities.
- ``wd_nchw``: WD EVA02 2026 Canary. White square, BICUBIC, RGB scaled to -1..1, NCHW.
- ``pixai``: PixAI v0.9. Stretched to the input size (BILINEAR), RGB -1..1, NCHW; the ``prediction``
  output holds probabilities.
- ``pixai_v1``: PixAI v1.0. Fitted into 1008 on a black square (BILINEAR), RGB -1..1, NCHW. Logits.
- ``siglip2``: CL Tagger v2. Stretched to 384 (BICUBIC), RGB -1..1, NCHW. Logits.
- ``cl``: CL Tagger v1. White square, BICUBIC, BGR -1..1, NCHW. Logits.

Every image is EXIF-oriented and its transparent pixels are composited over white first. Labels are
read into one category each: general, character, copyright, artist, meta, model, quality, rating.
General, meta and model tags use the general threshold; character, copyright and artist tags the
character threshold; rating and quality keep only their best label. A model's own fixed threshold
for a category (PixAI v1.0) replaces these.

Head detector (``deepghs/anime_head_detection``, YOLOv8): input ``[batch, 3, 640, 640]`` float32
RGB 0..1, output ``[batch, 5, anchors]`` (box cx/cy/w/h + one class score).
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
import queue
import re
import tempfile
import threading
import traceback
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

TAGGER_INPUT = 448
DETECTOR_INPUT = 640
PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
CATEGORIES = ("general", "character", "copyright", "artist", "meta", "model", "quality", "rating")
# The categories a caption gets unless others are chosen.
DEFAULT_CATEGORIES = ("general", "character")
_GENERAL_LIKE = {"general", "meta", "model"}
_SINGLE = {"rating", "quality"}
# WD CSV category ids.
_CSV_CATEGORIES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
    6: "quality",
    7: "model",
    9: "rating",
}
# PixAI v1.0 calls its artist group "style".
_CATEGORY_NAMES = {
    "copyrights": "copyright",
    "characters": "character",
    "style": "artist",
    "artists": "artist",
}
MAX_LABEL_BYTES = 64 * 1024 * 1024


# --------------------------------------------------------------------------- tagger
def _category(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_")
    name = _CATEGORY_NAMES.get(name, name)
    return name if name in CATEGORIES else "general"


def _checked(labels: list[tuple[str, str]], path: Path) -> tuple[tuple[str, str], ...]:
    if not labels or len(labels) > 500000:
        raise ValueError(f"{path.name} has no labels or too many")
    # An empty name is a placeholder slot (PixAI v0.9 has one); it keeps its place but is never written.
    if any(len(name) > 1000 for name, _ in labels):
        raise ValueError(f"{path.name} has an invalid tag name")
    if not any(category == "general" for _, category in labels):
        raise ValueError(f"{path.name} contains no general tags")
    return tuple(labels)


@lru_cache(maxsize=16)
def _labels_cached(path: str, kind: str, modified: int, size: int) -> tuple[tuple[str, str], ...]:
    del modified
    file = Path(path)
    if size > MAX_LABEL_BYTES:
        raise ValueError(f"{file.name} exceeds the supported 64 MiB size")
    labels: list[tuple[str, str]] = []
    if kind == "csv":
        with file.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"name", "category"}.issubset(reader.fieldnames):
                raise ValueError(f"{file.name} requires name and category columns")
            for row in reader:
                labels.append((row["name"].strip(), _CSV_CATEGORIES.get(int(row["category"]), "general")))
        return _checked(labels, file)
    data = json.loads(file.read_text(encoding="utf-8"))
    if kind == "pixai_json":
        # {num_classes, categories: [{name, offset, count, tags: [...]}]}, groups laid out by offset.
        for group in sorted(data["categories"], key=lambda item: item["offset"]):
            names = list(group["tags"])
            if group["offset"] != len(labels) or group["count"] != len(names):
                raise ValueError(f"{file.name} has overlapping or missing label groups")
            labels.extend((str(name).strip(), _category(group["name"])) for name in names)
        if len(labels) != data["num_classes"]:
            raise ValueError(f"{file.name} does not list num_classes labels")
    elif kind == "cl_vocabulary":
        # {idx_to_tag: {"0": tag, ...}, tag_to_category: {tag: "General", ...}}
        order = data["idx_to_tag"]
        pairs = enumerate(order) if isinstance(order, list) else sorted((int(k), v) for k, v in order.items())
        categories = data.get("tag_to_category") or {}
        for expected, (index, name) in enumerate(pairs):
            if index != expected:
                raise ValueError(f"{file.name} skips label {expected}")
            labels.append((str(name).strip(), _category(categories.get(name, "general"))))
    elif kind == "cl_mapping":
        # {"0": {"tag": ..., "category": "General"}, ...}
        for expected, (index, item) in enumerate(sorted((int(k), v) for k, v in data.items())):
            if index != expected:
                raise ValueError(f"{file.name} skips label {expected}")
            labels.append((str(item["tag"]).strip(), _category(item["category"])))
    else:
        raise ValueError(f"unknown label format {kind}")
    return _checked(labels, file)


def read_labels(path: str | Path, kind: str = "csv") -> tuple[tuple[str, str], ...]:
    """The model's labels in output order, each as (name, category)."""
    file = Path(path).expanduser().resolve()
    stat = file.stat()
    return _labels_cached(str(file), kind, stat.st_mtime_ns, stat.st_size)


def _flat_rgb(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
    background.alpha_composite(oriented)
    return background.convert("RGB")


def _white_square(rgb: Image.Image) -> Image.Image:
    side = max(rgb.size)
    if rgb.width == rgb.height:
        return rgb
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    return square


def _signed_chw(image: Image.Image, *, bgr: bool = False) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    chw = pixels.transpose(2, 0, 1)
    if bgr:
        chw = chw[::-1]
    return np.ascontiguousarray(((chw - 0.5) / 0.5)[None])


def prepare_tagger_input(image: Image.Image, spec: dict | None = None) -> np.ndarray:
    """One image as the tagger's input tensor (see the module docstring for each mode)."""
    spec = spec or {}
    mode, size = spec.get("preprocess", "wd"), int(spec.get("input_size", TAGGER_INPUT))
    rgb = _flat_rgb(image)
    if mode == "wd":
        square = _white_square(rgb)
        if square.size != (size, size):
            square = square.resize((size, size), Image.Resampling.LANCZOS)
        return np.ascontiguousarray(np.asarray(square, dtype=np.float32)[None, :, :, ::-1])
    if mode in ("wd_nchw", "cl"):
        square = _white_square(rgb).resize((size, size), Image.Resampling.BICUBIC)
        return _signed_chw(square, bgr=mode == "cl")
    if mode == "pixai":
        return _signed_chw(rgb.resize((size, size), Image.Resampling.BILINEAR))
    if mode == "siglip2":
        return _signed_chw(rgb.resize((size, size), Image.Resampling.BICUBIC))
    if mode == "pixai_v1":
        if rgb.size != (size, size):
            scale = min(size / rgb.height, size / rgb.width)
            fitted = rgb.resize(
                (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.Resampling.BILINEAR
            )
            rgb = Image.new("RGB", (size, size), (0, 0, 0))
            rgb.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
        return _signed_chw(rgb)
    raise ValueError(f"unknown tagger preprocessing {mode}")


def prepare_image(image: Image.Image, size: int = TAGGER_INPUT) -> np.ndarray:
    """WD v3 preprocessing: EXIF first, transparent pixels over white, white square, BGR 0..255."""
    return prepare_tagger_input(image, {"preprocess": "wd", "input_size": size})


_KAOMOJI_TAGS = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}


def readable(name: str, *, replace_underscore: bool = True, escape_parentheses: bool = False) -> str:
    if replace_underscore and name not in _KAOMOJI_TAGS:
        name = name.replace("_", " ")
    if escape_parentheses:
        name = re.sub(r"(?<!\\)([()])", lambda match: "\\" + match.group(0), name)
    return name


def select_tags(
    labels: tuple[tuple[str, str], ...],
    scores: np.ndarray,
    general: float,
    character: float,
    *,
    exclude: tuple[str, ...] = (),
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    fixed: dict[str, float] | None = None,
    replace_underscore: bool = True,
    escape_parentheses: bool = False,
) -> list[tuple[str, str, float]]:
    """The tags of one image above their thresholds as (tag, category, score), best first within
    each category; categories not asked for and excluded tags are dropped."""
    probabilities = np.asarray(scores).reshape(-1)
    if len(probabilities) != len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("the model output does not match its label table")
    blocked = {
        name.strip().replace("\\(", "(").replace("\\)", ")").replace("_", " ").casefold()
        for name in exclude
        if name.strip()
    }
    wanted, fixed = set(categories), fixed or {}
    picked: list[tuple[float, str, str]] = []
    best: dict[str, tuple[float, str]] = {}
    for (name, category), score in zip(labels, probabilities, strict=True):
        if category not in wanted or not name:
            continue
        value = float(score)
        tag = readable(name, replace_underscore=replace_underscore, escape_parentheses=escape_parentheses)
        if name.replace("\\(", "(").replace("\\)", ")").replace("_", " ").casefold() in blocked:
            continue
        if category in fixed:
            if value > fixed[category]:
                picked.append((value, tag, category))
        elif category in _SINGLE:
            if category not in best or value > best[category][0]:
                best[category] = (value, tag)
        elif value > (general if category in _GENERAL_LIKE else character):
            picked.append((value, tag, category))
    picked.extend((value, tag, category) for category, (value, tag) in best.items())
    return [
        (tag, category, value)
        for value, tag, category in sorted(picked, key=lambda item: (-item[0], item[1]))
    ]


def caption_tags(selected: list[tuple[str, str, float]], *, anima: bool = False) -> list[tuple[str, str]]:
    """Tags in caption order: people count, characters, works, artists (with Anima's @), the rest by
    confidence, then meta, model, quality and rating tags."""
    from .site_downloads import COUNT_TAG

    def rank(item: tuple[str, str, float]) -> int:
        tag, category, _ = item
        if category == "general":
            return 0 if COUNT_TAG.match(tag) else 4
        return {
            "character": 1,
            "copyright": 2,
            "artist": 3,
            "meta": 5,
            "model": 6,
            "quality": 7,
            "rating": 8,
        }[category]

    ordered = sorted(selected, key=rank)  # stable: confidence order stays inside each group
    return [
        (("@" + tag if anima and category == "artist" and not tag.startswith("@") else tag), category)
        for tag, category, _ in ordered
    ]


def _input_matches(shape: list, spec: dict) -> bool:
    size = int(spec.get("input_size", TAGGER_INPUT))
    dims = list(shape[1:])
    if len(dims) != 3:
        return False
    fits = lambda value, expected: not isinstance(value, int) or value in (expected, -1, 0)  # noqa: E731
    if spec.get("preprocess", "wd") == "wd":
        return fits(dims[0], size) and fits(dims[1], size) and fits(dims[2], 3)
    return fits(dims[0], 3) and fits(dims[1], size) and fits(dims[2], size)


# --------------------------------------------------------------------------- head detector
def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return overlap / np.maximum(area + areas - overlap, 1e-9)


def decode_detections(output: np.ndarray, confidence: float, iou_threshold: float) -> list[list[float]]:
    """YOLOv8 output ``[1, 5, anchors]`` -> boxes ``[x1, y1, x2, y2, score]`` in input pixels."""
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T  # [anchors, 4 + classes]
    boxes = raw[:, :4]
    scores = raw[:, 4:].max(axis=1)
    keep = scores > confidence
    boxes, scores = boxes[keep], scores[keep]
    if not len(boxes):
        return []
    xyxy = np.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        axis=1,
    )
    order = np.argsort(-scores)
    picked: list[int] = []
    remaining = list(order)
    while remaining:
        current = remaining.pop(0)
        picked.append(int(current))
        if remaining:
            overlaps = _iou(xyxy[current], xyxy[remaining])
            remaining = [
                index for index, value in zip(remaining, overlaps, strict=True) if value <= iou_threshold
            ]
    return [[*xyxy[index].tolist(), float(scores[index])] for index in picked]


def detector_regions(
    detections: list[list[float]],
    size: tuple[int, int],
    *,
    padding: float,
    feather: float,
) -> list[dict[str, Any]]:
    """Expand each detected box into the rectangle the mask clears, with its feather widths."""
    width, height = size
    regions = []
    for index, (x1, y1, x2, y2, score) in enumerate(detections):
        box_w, box_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        pad_x, pad_y = box_w * padding, box_h * padding
        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))
        right = min(width, int(np.ceil(x2 + pad_x)))
        bottom = min(height, int(np.ceil(y2 + pad_y)))
        if right - left < 1 or bottom - top < 1:
            continue
        regions.append(
            {
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom,
                "feather_x": round(box_w * feather),
                "feather_y": round(box_h * feather),
                "score": round(float(score), 4),
                "index": index,
            }
        )
    return regions


def rasterize_regions(regions: list[dict[str, Any]], size: tuple[int, int]) -> np.ndarray:
    """Grayscale mask: 255 = train, 0 = the cleared rectangle, a linear ramp outside it."""
    width, height = size
    mask = np.full((height, width), 255.0)
    if not regions:
        return mask
    ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
    xs = np.arange(width, dtype=np.float32).reshape(1, -1)
    for region in regions:
        # Outside distance in pixels: 0 inside the rectangle, growing outwards on each side.
        dx = np.maximum(np.maximum(region["x1"] - xs, xs - (region["x2"] - 1)), 0)
        dy = np.maximum(np.maximum(region["y1"] - ys, ys - (region["y2"] - 1)), 0)
        feather_x = max(1, int(region["feather_x"]))
        feather_y = max(1, int(region["feather_y"]))
        ramp = np.minimum(1.0, np.maximum(dx / feather_x, dy / feather_y))
        mask = np.minimum(mask, ramp * 255.0)
    return mask


def detector_prepare(
    image: Image.Image, size: int = DETECTOR_INPUT
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox onto a gray square (the Ultralytics convention); returns pixels, scale and padding."""
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(size / oriented.width, size / oriented.height)
    resized = oriented.resize(
        (max(1, round(oriented.width * scale)), max(1, round(oriented.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top = (size - resized.width) // 2, (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]), scale, (left, top)


def detector_restore(boxes: list[list[float]], scale: float, padding: tuple[int, int]) -> list[list[float]]:
    """Detection boxes back in the source image's own pixels."""
    left, top = padding
    return [
        [(x1 - left) / scale, (y1 - top) / scale, (x2 - left) / scale, (y2 - top) / scale, score]
        for x1, y1, x2, y2, score in boxes
    ]


# --------------------------------------------------------------------------- runtime
def available_providers() -> list[str]:
    import onnxruntime as ort

    ort.disable_telemetry_events()
    return list(ort.get_available_providers())


def _session(path: str | Path, provider: str, device_index: int = 0):
    import onnxruntime as ort

    ort.disable_telemetry_events()
    if provider == PROVIDERS["cuda"] and hasattr(ort, "preload_dlls"):
        # CUDA and cuDNN come from the nvidia-* wheels PyTorch installed in this environment.
        ort.preload_dlls()
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            f"{provider} is not installed; install the GPU build of ONNX Runtime in Environment settings or use the CPU"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    chosen = [(provider, {"device_id": device_index})] if provider == PROVIDERS["cuda"] else []
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=[*chosen, "CPUExecutionProvider"]
    )
    if provider not in session.get_providers():
        log.warning("%s failed to initialize; running this job on the CPU", provider)
    return session


def _worker(kind: str, images: list[str], options: dict, events: Any, working_dir: str) -> None:
    """Spawn entrypoint: heavy runtime state disappears when the process exits."""
    try:
        os.chdir(working_dir)
        provider = PROVIDERS[options.get("provider", "cpu")]
        if provider != PROVIDERS["cpu"] and provider not in available_providers():
            log.warning("%s is unavailable; using the CPU", provider)
            provider = PROVIDERS["cpu"]
        events.put({"type": "progress", "done": 0, "total": len(images), "message": "Loading model"})
        if kind == "tagger":
            spec = options["spec"]
            labels = read_labels(options["tags_path"], spec.get("labels", "csv"))
            session = _session(options["model_path"], provider, options.get("device_index", 0))
            inputs = session.get_inputs()
            if len(inputs) != 1 or not _input_matches(list(inputs[0].shape), spec):
                raise ValueError(
                    f"this tagger's input {inputs[0].shape if inputs else None} does not match its catalog entry"
                )
            outputs = session.get_outputs()
            if not outputs:
                raise ValueError("this tagger has no prediction output")
            names = [output.name for output in outputs]
            output = spec.get("output_name") if spec.get("output_name") in names else names[0]
            logits = spec.get("output") == "logits"
            results = []
            for index, path in enumerate(images):
                with Image.open(path) as image:
                    image.load()
                    pixels = prepare_tagger_input(image, spec)
                scores = np.asarray(session.run([output], {inputs[0].name: pixels})[0], dtype=np.float32)[0]
                if logits:
                    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
                selected = select_tags(
                    labels,
                    scores,
                    options.get("general_threshold", 0.35),
                    options.get("character_threshold", 0.85),
                    exclude=tuple(options.get("exclude_tags", ())),
                    categories=tuple(options.get("categories") or DEFAULT_CATEGORIES),
                    fixed=spec.get("fixed_thresholds"),
                    replace_underscore=options.get("replace_underscore", True),
                    escape_parentheses=options.get("escape_parentheses", False),
                )
                results.append(selected)
                events.put(
                    {"type": "progress", "done": index + 1, "total": len(images), "message": Path(path).name}
                )
            events.put({"type": "result", "tags": results})
            return
        session = _session(options["model_path"], provider, options.get("device_index", 0))
        inputs = session.get_inputs()
        if len(inputs) != 1 or list(inputs[0].shape[1:2]) != [3]:
            raise ValueError("this head detector needs a 3×640×640 input")
        outputs = session.get_outputs()
        if not outputs:
            raise ValueError("this head detector has no output")
        results = []
        for index, path in enumerate(images):
            try:
                with Image.open(path) as image:
                    image.load()
                    oriented = ImageOps.exif_transpose(image)
                    pixels, scale, offset = detector_prepare(oriented)
                    size = oriented.size
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                # One unreadable file leaves only itself out of the run.
                results.append({"regions": [], "size": None, "error": f"{type(exc).__name__}: {exc}"})
                events.put(
                    {"type": "progress", "done": index + 1, "total": len(images), "message": Path(path).name}
                )
                continue
            raw = session.run([outputs[0].name], {inputs[0].name: pixels})[0]
            boxes = detector_restore(
                decode_detections(raw, options.get("confidence", 0.413), options.get("iou_threshold", 0.7)),
                scale,
                offset,
            )
            boxes = [box for box in boxes if box[2] - box[0] >= 1 and box[3] - box[1] >= 1]
            results.append(
                {
                    "regions": detector_regions(
                        boxes,
                        size,
                        padding=options.get("padding", 0.10),
                        feather=options.get("feather", 0.03),
                    ),
                    "size": list(size),
                }
            )
            events.put(
                {"type": "progress", "done": index + 1, "total": len(images), "message": Path(path).name}
            )
        events.put({"type": "result", "detections": results})
    except BaseException as exc:  # noqa: BLE001 - the child reports every failure to its parent
        events.put(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        )


def _run(
    kind: str,
    images: list[str],
    options: dict,
    progress: Callable[[int, int, str], None],
    cancel: threading.Event,
) -> dict:
    if cancel.is_set():
        raise RuntimeError("cancelled")
    for key in ("model_path",):
        path = Path(options[key]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"model file is missing: {path}")
        options[key] = str(path)
    if kind == "tagger":
        tags = Path(options["tags_path"]).expanduser().resolve()
        if not tags.is_file():
            raise ValueError(f"label table is missing: {tags}")
        read_labels(tags, options["spec"].get("labels", "csv"))
        options["tags_path"] = str(tags)
    context = multiprocessing.get_context("spawn")
    workspace = tempfile.TemporaryDirectory(prefix="ypuddin-vision-worker-")
    events = context.Queue()
    process = context.Process(
        target=_worker, args=(kind, images, options, events, workspace.name), daemon=True
    )
    started = False
    try:
        process.start()
        started = True
        while True:
            if cancel.is_set():
                raise RuntimeError("cancelled")
            try:
                message = events.get(timeout=0.1)
            except queue.Empty:
                if process.is_alive():
                    continue
                try:
                    message = events.get(timeout=0.2)  # the feeder may outlive the worker briefly
                except queue.Empty:
                    raise RuntimeError(
                        f"inference process exited without a result (exit code {process.exitcode})"
                    ) from None
            if message["type"] == "progress":
                progress(message["done"], message["total"], message["message"])
            elif message["type"] == "result":
                if cancel.is_set():
                    raise RuntimeError("cancelled")
                return message
            else:
                if message.get("traceback"):
                    log.debug("vision worker failed: %s", message["traceback"])
                raise RuntimeError(message["message"])
    finally:
        if started:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            process.close()
        events.close()
        events.cancel_join_thread()
        workspace.cleanup()


def tag_images(
    images: list[str],
    *,
    model_path: str | Path,
    tags_path: str | Path,
    spec: dict | None = None,
    general: float = 0.35,
    character: float = 0.85,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    exclude: tuple[str, ...] = (),
    anima: bool = False,
    provider: str = "cpu",
    device_index: int = 0,
    replace_underscore: bool = True,
    escape_parentheses: bool = False,
    progress: Callable[[int, int, str], None] = lambda *_: None,
    cancel: threading.Event | None = None,
) -> list[list[tuple[str, str]]]:
    """Tag each image; returns, in the order given, each image's tags as (tag, category) in caption
    order. ``spec`` is the tagger's catalog entry (WD v3 when omitted)."""
    entry = spec or {}
    options = {
        "model_path": str(model_path),
        "tags_path": str(tags_path),
        "spec": {
            key: entry.get(key, default)
            for key, default in (
                ("preprocess", "wd"),
                ("input_size", TAGGER_INPUT),
                ("labels", "csv"),
                ("output", "probability"),
                ("output_name", None),
                ("fixed_thresholds", {}),
            )
        },
        "general_threshold": general,
        "character_threshold": character,
        "categories": list(categories),
        "exclude_tags": list(exclude),
        "replace_underscore": replace_underscore,
        "escape_parentheses": escape_parentheses,
        "provider": provider,
        "device_index": device_index,
    }
    result = _run("tagger", [str(p) for p in images], options, progress, cancel or threading.Event())
    return [caption_tags([tuple(item) for item in tags], anima=anima) for tags in result["tags"]]


def detect_heads(
    images: list[str],
    *,
    model_path: str | Path,
    confidence: float = 0.413,
    iou_threshold: float = 0.7,
    padding: float = 0.10,
    feather: float = 0.03,
    provider: str = "cpu",
    device_index: int = 0,
    progress: Callable[[int, int, str], None] = lambda *_: None,
    cancel: threading.Event | None = None,
) -> list[dict[str, Any]]:
    """Detect heads in each image; returns one ``{"regions": [...], "size": [w, h]}`` per image."""
    options = {
        "model_path": str(model_path),
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "padding": padding,
        "feather": feather,
        "provider": provider,
        "device_index": device_index,
    }
    return _run("detector", [str(p) for p in images], options, progress, cancel or threading.Event())[
        "detections"
    ]


def model_info(path: str | Path) -> dict[str, Any]:
    """Input/output shapes of an ONNX file, for checking a model before a job uses it."""
    import onnxruntime as ort

    ort.disable_telemetry_events()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()],
        "metadata": dict(session.get_modelmeta().custom_metadata_map or {}),
        "providers": list(ort.get_available_providers()),
    }


@lru_cache(maxsize=8)
def _runtime_probe(identity: tuple) -> dict[str, Any]:
    del identity  # native-library metadata forms the cache key; ONNX Runtime never loads in the service
    import subprocess
    import sys

    code = (
        "import json, importlib.metadata as m, onnxruntime as o; o.disable_telemetry_events()\n"
        "names = [d for d in ('onnxruntime-gpu', 'onnxruntime') if any(x.metadata['Name'].lower() == d for x in m.distributions())]\n"
        "print(json.dumps({'version': o.__version__, 'package': names[0] if names else None, 'providers': o.get_available_providers()}))"
    )
    with tempfile.TemporaryDirectory(prefix="ypuddin-ort-probe-") as directory:
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=directory, capture_output=True, text=True, timeout=30
        )
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or "ONNX Runtime probe failed")[-1600:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def runtime_status() -> dict[str, Any]:
    """Whether ONNX Runtime can run here and with which providers, probed in a child process."""
    import importlib.util

    try:
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("onnxruntime")
        if not spec or not spec.origin:
            raise ModuleNotFoundError("ONNX Runtime is not installed")
        origin = Path(spec.origin)
        assets = [origin, *sorted((origin.parent / "capi").glob("onnxruntime_pybind11_state*"))]
        identity = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in assets)
        probe = _runtime_probe(identity)
    except Exception as exc:  # noqa: BLE001 - any failure means the runtime cannot be used
        return {
            "available": False,
            "package": None,
            "version": None,
            "providers": [],
            "error": str(exc)[-1600:],
        }
    names = probe["providers"]
    return {
        "available": True,
        "package": probe.get("package"),
        "version": probe["version"],
        "providers": [name for name, provider in PROVIDERS.items() if provider in names],
        "error": None,
    }
