"""JSONL adapter for KohakuGBC's Apache-2.0 machine API.

Upstream: KohakuBlueleaf/KohakuGBC, d90f360dbf561f63270522e44fa0e74ebed59c12.
The official node program, detector, embedder and reply parsers run unchanged. Only chat
transport is supplied here; requests go back to the owning service without credentials.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
from pathlib import Path


class Protocol:
    def __init__(self):
        self.input, self.output = sys.stdin, sys.stdout
        self.lock = threading.Lock()
        self.serial = 0
        self.errors = []
        self.work = None

    def send(self, message):
        self.output.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.output.flush()

    def receive(self):
        line = self.input.readline(8 * 1024 * 1024 + 1)
        if not line:
            raise EOFError("owning service disconnected")
        if len(line) > 8 * 1024 * 1024:
            raise ValueError("request exceeds size limit")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("request must be an object")
        return value

    def chat(self, request):
        from kohakugbc.nodes.specs import chat, parse

        # entity_batch=1 keeps the official image/entity/relation/composition requests single-image.
        pictures = [(label, picture) for label, picture in request["images"] if picture is not None]
        if len(pictures) != 1:
            raise ValueError("expected one GBC request image")
        with self.lock:
            try:
                self.serial += 1
                image = self.work / f"request-{self.serial}.jpg"
                image.write_bytes(pictures[0][1]["jpeg"])
                schema_name = request.get("schema")
                schema = chat.schema(schema_name) if schema_name else None
                prompt = "\n\n".join(part for part in (
                    request.get("system"), pictures[0][0], request.get("query"),
                    "Return only JSON matching this schema:\n" + json.dumps(schema) if schema else None,
                ) if part)
                self.send({"type": "request", "id": self.serial, "image": str(image), "prompt": prompt})
                reply = self.receive()
                if reply.get("type") != "reply" or reply.get("id") != self.serial:
                    raise ValueError("unexpected reply order")
                text = reply.get("text")
                if not isinstance(text, str) or not parse.reply_ok(text, schema_name, schema):
                    raise ValueError("VLM reply does not match the GBC request schema")
                if schema_name == "image.json" and not parse.image(text).get("descs"):
                    raise ValueError("VLM image reply contains no descriptions")
                return text
            except Exception as exc:
                self.errors.append(str(exc))
                raise


def _parts(args, protocol):
    from kohakugbc.engines.registry import Slot, entry
    from kohakugbc.engines.torch.entries import detector, embedder
    from kohakugbc.machine.build import build_parts
    from kohakugbc.nodes import library, registry
    from togeari.engines.python import PythonCallEngine

    # A dedicated process already supplies isolation; no optional engine-host CLI is needed.
    entry("detect", "ypuddin_local")(detector)
    entry("embed", "ypuddin_local")(embedder)

    @entry("vlm", "ypuddin_callback")
    def callback_slot(settings, shared):
        return Slot(units=[PythonCallEngine(workers=1, mode="thread")], call=protocol.chat)

    # The upstream KIR intentionally turns failed chat requests into empty replies. Keep its
    # behavior, but reject the resulting graph if any request failed during this image.
    def record_failures(original):
        def no_reply(err):
            protocol.errors.append(str(err))
            return original(err)

        return no_reply

    for index, (fn, outputs, flags) in enumerate(registry.NODES):
        if fn.__name__ == "no_reply":
            registry.NODES[index] = (record_failures(fn), outputs, flags)
            break
    else:
        raise RuntimeError("GBC no_reply node is missing")
    del library  # Import above registers all official plain nodes before the wrapper is installed.
    return build_parts({
        "engines": {
            "vlm": {"entry": "ypuddin_callback"},
            "detect": {"entry": "ypuddin_local", "model": args.detector,
                       "devices": [args.device], "dtype": "float32", "max_batch": 1},
            "embed": {"entry": "ypuddin_local", "model": args.embedder, "impl": "st",
                      "devices": ["cpu"], "dtype": "float32", "max_batch": 8},
            "cpu": {"entry": "python", "workers": 2, "mode": "thread"},
        },
        "machine": {"max_jobs": 1, "vlm_waiting": 1, "window": 16, "store": "memory",
                    "start_timeout_s": 120},
    })


class Capture:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def close(self):
        pass


def _analyze(parts, command, protocol):
    from PIL import Image, ImageOps
    from togeari import SourceItem

    protocol.errors.clear()
    source = Path(command["image"])
    # The graph's coordinate system and the VLM see the same EXIF-oriented, opaque image.
    with Image.open(source) as image:
        oriented = ImageOps.exif_transpose(image)
        if oriented.mode in {"RGBA", "LA", "PA"} or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            oriented = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
        oriented = oriented.convert("RGB")
        picture = protocol.work / f"image-{command['id']}.png"
        oriented.save(picture)
    controls = {"min_nodes": 1, "max_nodes": command["max_nodes"], "max_depth": command["max_depth"],
                "max_rounds": command["max_depth"] + 1, "relational": "both", "entity_batch": 1}
    sink = Capture()
    parts.machine.run([SourceItem(str(command["id"]), {"image": {"path": str(picture), "key": source.name},
                                                       "controls": controls})], sink)
    if protocol.errors:
        raise RuntimeError("; ".join(protocol.errors)[:3000])
    if len(sink.records) != 1 or not sink.records[0].ok:
        raise RuntimeError(str(sink.records[0].failure if sink.records else "GBC produced no record"))
    graph = json.loads(sink.records[0].outputs["graph"])
    if not graph.get("vertices"):
        raise RuntimeError("GBC produced no described vertices")
    graph["img_path"] = str(source)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--detector")
    parser.add_argument("--embedder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    protocol = Protocol()
    try:
        # Library progress and warnings must never be mistaken for protocol messages.
        with contextlib.redirect_stdout(sys.stderr):
            if args.check:
                from importlib.metadata import version

                from kohakugbc.engines.torch.entries import detector, embedder  # noqa: F401
                from kohakugbc.machine.build import build_parts  # noqa: F401

                for name, expected in (("kohakugbc", "0.1.0"), ("kohakugbc-data", "0.2.0"), ("togeari", "0.3.0")):
                    if version(name) != expected:
                        raise ValueError(f"{name}: expected {expected}, found {version(name)}")
                return 0
            if not args.work or not args.detector or not args.embedder:
                raise ValueError("missing runtime paths")
            protocol.work = args.work.resolve()
            protocol.work.mkdir(parents=True, exist_ok=True)
            parts = _parts(args, protocol)
            with parts.machine:
                from importlib.metadata import version

                protocol.send({"type": "ready", "pid": os.getpid(), "detector_device": args.device,
                               "embedder_device": "cpu", "detector_dtype": "float32", "embedder_dtype": "float32",
                               "versions": {name: version(name) for name in (
                                   "kohakugbc", "kohakugbc-data", "togeari", "torch", "transformers",
                               )}})
                while True:
                    command = protocol.receive()
                    if command.get("type") == "close":
                        break
                    if command.get("type") != "analyze":
                        raise ValueError("expected analyze command")
                    graph = _analyze(parts, command, protocol)
                    protocol.send({"type": "graph", "id": command["id"], "graph": graph})
        return 0
    except Exception as exc:
        protocol.send({"type": "error", "message": str(exc)[:3000]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
