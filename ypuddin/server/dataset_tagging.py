"""Local WD14/WDv3 inference in a cancellable process; caption writes belong to the pipeline.

Input contract verified against the model author's ONNX example:
https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py
White alpha composite, centered square padding, bicubic resize, NHWC BGR float32
in the original 0..255 range. Rating labels are never appended to training captions.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = 448
PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}


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


def prepare_image(image: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """The model's preprocessing, including EXIF orientation and transparent images."""
    oriented = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
    background.alpha_composite(oriented)
    rgb = background.convert("RGB")
    side = max(rgb.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    if side != size:
        square = square.resize((size, size), Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(square, dtype=np.float32)[None, :, :, ::-1])


def select_tags(
    labels: tuple[tuple[str, int], ...], scores: np.ndarray, general: float, character: float
) -> str:
    probabilities = np.asarray(scores).reshape(-1)
    if len(probabilities) != len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("ONNX output does not match selected_tags.csv")
    selected = []
    for (name, category), score in zip(labels, probabilities, strict=True):
        threshold = general if category == 0 else character if category == 4 else None
        if threshold is not None and float(score) > threshold:
            # Keep emoticons intact, while ordinary WD labels become readable caption tags.
            if not re.fullmatch(r"[0-9oOxXuU=^<>@|+.;()\-_]+", name):
                name = name.replace("_", " ")
            selected.append((float(score), name))
    return ", ".join(name for _, name in sorted(selected, key=lambda item: (-item[0], item[1])))


@lru_cache(maxsize=8)
def _runtime_probe(identity: tuple) -> dict[str, Any]:
    del identity  # Native-library metadata forms the cache key; never load ORT into the server.
    code = "import json, onnxruntime as o; o.disable_telemetry_events(); print(json.dumps({'version':o.__version__,'providers':o.get_available_providers()}))"
    with tempfile.TemporaryDirectory(prefix="ypuddin-ort-probe-") as directory:
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=directory, capture_output=True, text=True, timeout=15
        )
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or "runtime probe failed")[-1600:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def runtime_info() -> dict[str, Any]:
    try:
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("onnxruntime")
        if not spec or not spec.origin:
            raise ModuleNotFoundError("install the optional ONNX Runtime package in Environment settings")
        origin = Path(spec.origin)
        assets = [origin, *sorted((origin.parent / "capi").glob("onnxruntime_pybind11_state*"))]
        identity = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in assets)
        runtime = _runtime_probe(identity)
        names = runtime["providers"]
        return {
            "runtime_available": True,
            "runtime_version": runtime["version"],
            "runtime_providers": names,
            "providers": [name for name, provider in PROVIDERS.items() if provider in names],
            "runtime_error": None,
        }
    except Exception as exc:
        return {
            "runtime_available": False,
            "runtime_version": None,
            "runtime_providers": [],
            "providers": [],
            "runtime_error": f"ONNX Runtime is unavailable: {exc}",
        }


def local_models(models_dir: Path) -> list[dict[str, str]]:
    root = models_dir / "tagger"
    if not root.is_dir():
        return []
    choices = []
    for model in sorted(root.rglob("model.onnx")):
        tags = model.with_name("selected_tags.csv")
        if model.is_file() and tags.is_file() and not model.is_symlink() and not tags.is_symlink():
            choices.append(
                {
                    "name": str(model.parent.relative_to(root)),
                    "path": str(model.parent),
                    "model_path": str(model),
                    "tags_path": str(tags),
                }
            )
        if len(choices) >= 100:
            break
    return choices


def _worker(images: list[str], options: dict, events: Any, working_dir: str | None = None) -> None:
    """Spawn entrypoint. Heavy runtime state disappears when the process exits."""
    try:
        if working_dir:
            os.chdir(working_dir)  # Only this isolated process changes cwd; its parent removes the directory.
        import onnxruntime as ort

        ort.disable_telemetry_events()
        provider = PROVIDERS[options.get("provider", "cpu")]
        if provider not in ort.get_available_providers():
            raise RuntimeError(
                f"{provider} is not installed; select CPU or configure a compatible ONNX Runtime GPU build"
            )
        events.put(
            {
                "type": "progress",
                "done": 0,
                "total": len(images),
                "message": f"Loading local WD14 model with {provider}",
            }
        )
        labels = read_labels(options["tags_path"])
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = min(4, os.cpu_count() or 1)
        settings.inter_op_num_threads = 1
        settings.log_severity_level = 3
        session = ort.InferenceSession(options["model_path"], sess_options=settings, providers=[provider])
        if provider not in session.get_providers():
            raise RuntimeError(
                f"{provider} failed to initialize; check the runtime's CUDA/cuDNN dependencies or select CPU"
            )
        inputs = session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].type != "tensor(float)"
            or list(inputs[0].shape[1:]) != [INPUT_SIZE, INPUT_SIZE, 3]
        ):
            raise ValueError("this tagger supports WD14 NHWC 448×448×3 float32 ONNX input")
        outputs = session.get_outputs()
        if not outputs:
            raise ValueError("ONNX model has no prediction output")
        captions = []
        for index, path in enumerate(images):
            with Image.open(path) as image:
                image.load()
                pixels = prepare_image(image)
            scores = session.run([outputs[0].name], {inputs[0].name: pixels})[0]
            captions.append(
                select_tags(
                    labels,
                    scores[0],
                    options.get("general_threshold", 0.35),
                    options.get("character_threshold", 0.85),
                )
            )
            events.put(
                {"type": "progress", "done": index + 1, "total": len(images), "message": Path(path).name}
            )
        events.put({"type": "result", "captions": captions})
    except BaseException as exc:
        events.put(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        )


def generate(
    images: list[dict], options: dict, progress: Callable[[int, int, str], None], cancel: threading.Event
) -> list[str]:
    """Run local inference, returning drafts only. Cancellation can interrupt model loading."""
    if cancel.is_set():
        raise RuntimeError("cancelled")
    model = Path(options["model_path"]).expanduser().resolve()
    tags = Path(options["tags_path"]).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise ValueError("select an existing local model.onnx file")
    if not tags.is_file():
        raise ValueError("select the matching selected_tags.csv file")
    read_labels(tags)
    context = multiprocessing.get_context("spawn")
    workspace = tempfile.TemporaryDirectory(prefix="ypuddin-wd14-worker-")
    events = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            [str(image["path"]) for image in images],
            {**options, "model_path": str(model), "tags_path": str(tags)},
            events,
            workspace.name,
        ),
        daemon=True,
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
                if not process.is_alive():
                    # The feeder can briefly outlive the worker's final Python instruction.
                    try:
                        message = events.get(timeout=0.2)
                    except queue.Empty:
                        raise RuntimeError(
                            f"tagging worker exited without a result (exit code {process.exitcode})"
                        ) from None
                else:
                    continue
            if message["type"] == "progress":
                progress(message["done"], message["total"], message["message"])
            elif message["type"] == "result":
                if cancel.is_set():
                    raise RuntimeError("cancelled")
                return message["captions"]
            else:
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
