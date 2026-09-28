"""Local ONNX vision models: WD v3 taggers and the anime head detector.

Inference runs in a spawned child process so the service never loads ONNX Runtime itself and every
job starts from a clean runtime (a CUDA provider that fails falls back to the CPU for the rest of
the job). Only the ONNX graph and its label table are read; no model repository code is executed.

Tagger (WD v3): input ``[batch, 448, 448, 3]`` float32 NHWC in BGR 0..255, output ``[batch, tags]``
probabilities. Labels come from ``selected_tags.csv`` (name, category; 0 general, 4 character,
9 rating). Rating labels are never written into a caption.

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
# Rating labels (category 9) describe the dataset, not the image content, and never train.
RATING_CATEGORY = 9
CHARACTER_CATEGORY = 4
# Kept as written: these WD labels are emoticons and must not gain spaces.
_EMOTICON = re.compile(r"[0-9oOxXuU=^<>@|+.;()\-_]+")


# --------------------------------------------------------------------------- tagger
@lru_cache(maxsize=16)
def _labels_cached(path: str, modified: int, size: int) -> tuple[tuple[str, int], ...]:
    del modified
    if size > 20 * 1024 * 1024:
        raise ValueError("selected_tags.csv exceeds the supported 20 MiB size")
    labels = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"name", "category"}.issubset(reader.fieldnames):
            raise ValueError("selected_tags.csv requires name and category columns")
        for row in reader:
            name = row["name"].strip()
            category = int(row["category"])
            if not name or len(name) > 1000 or len(labels) >= 100000:
                raise ValueError("invalid WD14 tag names or too many labels")
            labels.append((name, category))
    if not labels or not any(category == 0 for _, category in labels):
        raise ValueError("selected_tags.csv contains no general tags")
    return tuple(labels)


def read_labels(path: str | Path) -> tuple[tuple[str, int], ...]:
    file = Path(path).expanduser().resolve()
    stat = file.stat()
    return _labels_cached(str(file), stat.st_mtime_ns, stat.st_size)


def prepare_image(image: Image.Image, size: int = TAGGER_INPUT) -> np.ndarray:
    """WD v3 preprocessing: EXIF first, transparent pixels over white, white square, BGR 0..255."""
    oriented = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
    background.alpha_composite(oriented)
    rgb = background.convert("RGB")
    side = max(rgb.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    if side != size:
        square = square.resize((size, size), Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(square, dtype=np.float32)[None, :, :, ::-1])


def select_tags(
    labels: tuple[tuple[str, int], ...],
    scores: np.ndarray,
    general: float,
    character: float,
    *,
    exclude: tuple[str, ...] = (),
) -> str:
    """Threshold, order and join the tags of one image; ratings and excluded tags are dropped."""
    probabilities = np.asarray(scores).reshape(-1)
    if len(probabilities) != len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("ONNX output does not match selected_tags.csv")
    blocked = {name.strip().replace("_", " ").casefold() for name in exclude if name.strip()}
    selected = []
    for (name, category), score in zip(labels, probabilities, strict=True):
        threshold = general if category == 0 else character if category == CHARACTER_CATEGORY else None
        if threshold is None or float(score) <= threshold:
            continue
        readable = name if _EMOTICON.fullmatch(name) else name.replace("_", " ")
        if readable.casefold() in blocked:
            continue
        selected.append((float(score), readable))
    return ", ".join(name for _, name in sorted(selected, key=lambda item: (-item[0], item[1])))


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
            labels = read_labels(options["tags_path"])
            session = _session(options["model_path"], provider, options.get("device_index", 0))
            inputs = session.get_inputs()
            if len(inputs) != 1 or list(inputs[0].shape[1:]) != [TAGGER_INPUT, TAGGER_INPUT, 3]:
                raise ValueError("this tagger needs the WD v3 NHWC 448×448×3 input")
            outputs = session.get_outputs()
            if not outputs:
                raise ValueError("this tagger has no prediction output")
            results, characters = [], []
            names = {
                name if _EMOTICON.fullmatch(name) else name.replace("_", " ")
                for name, category in labels
                if category == CHARACTER_CATEGORY
            }
            for index, path in enumerate(images):
                with Image.open(path) as image:
                    image.load()
                    pixels = prepare_image(image)
                scores = session.run([outputs[0].name], {inputs[0].name: pixels})[0]
                caption = select_tags(
                    labels,
                    np.asarray(scores)[0],
                    options.get("general_threshold", 0.35),
                    options.get("character_threshold", 0.85),
                    exclude=tuple(options.get("exclude_tags", ())),
                )
                results.append(caption)
                characters.append([tag for tag in caption.split(", ") if tag in names])
                events.put(
                    {"type": "progress", "done": index + 1, "total": len(images), "message": Path(path).name}
                )
            events.put({"type": "result", "captions": results, "characters": characters})
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
            with Image.open(path) as image:
                image.load()
                oriented = ImageOps.exif_transpose(image)
                pixels, scale, offset = detector_prepare(oriented)
                size = oriented.size
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
        read_labels(tags)
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
    general: float = 0.35,
    character: float = 0.85,
    exclude: tuple[str, ...] = (),
    provider: str = "cpu",
    device_index: int = 0,
    progress: Callable[[int, int, str], None] = lambda *_: None,
    cancel: threading.Event | None = None,
    characters: bool = False,
) -> list[str] | list[tuple[str, list[str]]]:
    """Tag each image; returns one caption per image, in the order given.

    With ``characters``, each entry is ``(caption, character tags in it)``.
    """
    options = {
        "model_path": str(model_path),
        "tags_path": str(tags_path),
        "general_threshold": general,
        "character_threshold": character,
        "exclude_tags": list(exclude),
        "provider": provider,
        "device_index": device_index,
    }
    result = _run("tagger", [str(p) for p in images], options, progress, cancel or threading.Event())
    if characters:
        return list(zip(result["captions"], result["characters"], strict=True))
    return result["captions"]


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
