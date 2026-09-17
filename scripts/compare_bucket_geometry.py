"""Read-only geometry comparison. No model or image-pixel quality claims."""

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("images", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--reference-root", type=Path, default=REPO.parent)
args = parser.parse_args()
ROOT = args.reference_root
sys.path.insert(0, str(REPO))
from ypuddin.data.buckets import BUCKET_POLICY, BucketManager, fit_crop, fit_pad  # noqa: E402

old = types.ModuleType("old_buckets")
sys.modules[old.__name__] = old
exec(
    subprocess.check_output(["git", "show", "a5873d8:ypuddin/data/buckets.py"], cwd=REPO).decode(),
    old.__dict__,
)
als_path = ROOT / "AnimaLoraStudio/runtime/training/dataset.py"
als_commit = "3d9d2e86045b879cd19c01ac4aa2337f45a283ab"
als_source = subprocess.check_output(
    ["git", "show", f"{als_commit}:runtime/training/dataset.py"], cwd=ROOT / "AnimaLoraStudio"
).decode()
node = next(
    n for n in ast.parse(als_source).body if isinstance(n, ast.ClassDef) and n.name == "BucketManager"
)
ns = {"math": math}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(als_path), "exec"), ns)
ALS = ns["BucketManager"]


def measure(sizes, name, fit="crop", no_upscale=False, base=1024):
    manager = (
        ALS(base_reso=base)
        if name.startswith("als")
        else (old.BucketManager if name == "old" else BucketManager)([base], no_upscale=no_upscale)
    )
    rows = []
    start = time.perf_counter()
    for w, h in sizes:
        if name.startswith("als"):
            bw, bh = manager.get_bucket(w, h)
            if name == "als":
                scale = max(bw / w, bh / h)
                rw, rh = int(w * scale), int(h * scale)
            else:
                rw, rh, *_ = fit_crop(w, h, bw, bh)
        else:
            b = manager.assign(w, h, base, **({"image_fit": fit} if name == "new" else {}))
            bw, bh = b.key
            rw, rh, *_ = (
                fit_crop(w, h, bw, bh)
                if fit == "crop"
                else fit_pad(w, h, bw, bh, max_scale=1 if no_upscale else None)
            )
        loss = 1 - (min(bw, rw) * min(bh, rh) / (rw * rh) if fit == "crop" else rw * rh / (bw * bh))
        rows.append({"input": [w, h], "bucket": [bw, bh], "resize": [rw, rh], "loss": loss})
    elapsed = time.perf_counter() - start
    return {
        "count": len(rows),
        "mean_loss_percent": sum(r["loss"] for r in rows) * 100 / len(rows),
        "max_loss_percent": max(r["loss"] for r in rows) * 100,
        "upscaled": sum(r["resize"][0] > r["input"][0] or r["resize"][1] > r["input"][1] for r in rows),
        "downscaled": sum(r["resize"][0] < r["input"][0] or r["resize"][1] < r["input"][1] for r in rows),
        "unchanged_size": sum(r["resize"] == r["input"] for r in rows),
        "mean_canvas_pixels": sum(r["bucket"][0] * r["bucket"][1] for r in rows) / len(rows),
        "used_bucket_count": len(set(tuple(r["bucket"]) for r in rows)),
        "selection_seconds": elapsed,
        "rows": rows,
    }


def compare(sizes):
    output = {}
    for fit in ("crop", "pad"):
        for no_upscale in (False, True):
            entries = {}
            for name in (
                ("old", "als", "als_shared_transform", "new")
                if fit == "crop" and not no_upscale
                else ("old", "new")
            ):
                entries[name] = measure(sizes, name, fit, no_upscale)
            for name, entry in entries.items():
                if name == "new":
                    entry["relative_to_old"] = dict(
                        Counter(
                            "better"
                            if n["loss"] < o["loss"] - 1e-12
                            else "worse"
                            if n["loss"] > o["loss"] + 1e-12
                            else "equal"
                            for n, o in zip(entry["rows"], entries["old"]["rows"], strict=True)
                        )
                    )
            for entry in entries.values():
                del entry["rows"]
            output[f"{fit}/no_upscale={no_upscale}"] = entries
    return output


sizes = []
for path in sorted(args.images.rglob("*")):
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} or path.stem.endswith(".mask"):
        continue
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(274, 1) in {5, 6, 7, 8}:
            w, h = h, w
        sizes.append((w, h))
if not sizes:
    parser.error("No input images found")
synthetic = [(w, 1000) for w in range(500, 2001)]
synthetic += [(h, w) for w, h in synthetic]
result = {
    "policy": BUCKET_POLICY,
    "old_commit": "a5873d825a60c3dba9543cd6570f83af982f354f",
    "als_commit": als_commit,
    "config": {"base": 1024, "step": 64, "aspect_ratio_limit": 2, "area_tolerance": 0.1, "align": 16},
    "candidate_count": {
        "old": len(old.BucketManager([1024]).buckets[1024]),
        "als": len(ALS().buckets),
        "new": len(BucketManager([1024]).buckets[1024]),
    },
    "input_sizes_sha256": hashlib.sha256(json.dumps(sizes).encode()).hexdigest(),
    "input_size_histogram": {f"{w}x{h}": count for (w, h), count in sorted(Counter(sizes).items())},
    "provided_images": compare(sizes),
    "synthetic_3002_images": compare(synthetic),
    "example_522x1000": {name: measure([(522, 1000)], name) for name in ("old", "als", "new")},
    "limitations": [
        "Geometry only; no trained-model quality or GPU throughput comparison.",
        "ALS uses its original absolute-ratio selector and truncating crop resize; YPuddin uses its production rounded geometry.",
        "als_shared_transform applies YPuddin pixel rounding to ALS bucket choices, isolating the selector from resize differences.",
        "No-upscale and padding rows compare only YPuddin; ALS is not presented as implementing padding.",
        "Selection CPU timings include geometry collection; not training throughput.",
        "Input image files were read only for dimensions and EXIF orientation; no pixels, captions or filenames are in this result.",
    ],
}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "candidate_count": result["candidate_count"],
            "provided_images": result["provided_images"],
            "synthetic": result["synthetic_3002_images"],
        },
        indent=2,
    )
)
