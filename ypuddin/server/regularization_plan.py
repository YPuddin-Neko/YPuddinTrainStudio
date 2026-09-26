"""Read-only plans for one prior image per training image; no caption rewriting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ypuddin.data.captions import read_editable_caption
from ypuddin.data.index import caption_target, content_hash, iter_images

from .errors import ApiError


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize_tag(tag: str) -> str:
    return " ".join(tag.replace("_", " ").casefold().split())


def training_sources(context, pid: str, vid: str, config: dict, source_ids=()) -> list[dict]:
    """The version's training folders, not its regularization ones, with their caption settings."""
    context.resolve_version(pid, vid)
    registry = {
        str(Path(row["path"]).expanduser().resolve()): row["id"]
        for row in context.db.fetchall("SELECT id,path FROM datasets WHERE version_id=?", (vid,))
    }
    sources = []
    seen_roots = set()
    reg_root = context.reg_dir(pid, vid).resolve()
    for source in config.get("dataset", {}).get("sources", []):
        root = Path(source["path"]).expanduser().resolve()
        if source.get("is_reg") or root.is_relative_to(reg_root) or str(root) in seen_roots:
            continue
        if not context.is_allowed(root):
            raise ApiError("训练来源不在允许的目录内", status=403, code="regularization.path")
        seen_roots.add(str(root))
        sid = registry.get(str(root)) or f"path_{_digest(str(root))[:24]}"
        sources.append(
            {
                "id": sid,
                "path": str(root),
                "name": root.name,
                "caption_ext": source.get("caption_ext", "auto"),
                "trigger_word": (source.get("caption") or {}).get("trigger_word"),
            }
        )
    if set(source_ids) - {source["id"] for source in sources}:
        raise ApiError("所选训练目录已变化，请重新选择范围", status=409, code="regularization.sources")
    return sources


def training_plan(context, pid: str, vid: str, request, config: dict) -> tuple[dict, list[dict]]:
    sources = training_sources(context, pid, vid, config, request.source_ids)
    chosen = set(request.source_ids)
    reg_root = context.reg_dir(pid, vid).resolve()

    existing = set()
    for row in context.db.fetchall(
        "SELECT path FROM regularization_operations WHERE project_id=? AND version_id=? AND status='completed'",
        (pid, vid),
    ):
        if not row["path"]:
            continue
        batch = Path(row["path"]).resolve()
        if not batch.is_relative_to(reg_root):
            continue
        manifest = batch / "manifest.json"
        try:
            if manifest.stat().st_size > 16 * 1024 * 1024:
                continue
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                provenance = entry.get("training_source")
                filename = entry.get("file")
                if not isinstance(provenance, dict) or not isinstance(filename, str):
                    continue
                image = (batch / filename).resolve()
                if (
                    image.is_relative_to(batch)
                    and image.is_file()
                    and isinstance(provenance.get("identity"), str)
                ):
                    existing.add(provenance["identity"])
        except (OSError, ValueError):
            continue

    excluded = {normalize_tag(tag) for tag in request.excluded_tags if normalize_tag(tag)}
    frequencies = {}
    candidates = []
    counts = {
        "source_images": 0,
        "existing_images": 0,
        "missing_captions": 0,
        "invalid_captions": 0,
        "empty_after_exclusion": 0,
    }
    signature_rows = []
    caption_directories = {}
    for source in sources:
        if chosen and source["id"] not in chosen:
            continue
        root = Path(source["path"])
        if not root.is_dir():
            raise ApiError("训练目录不存在，请先检查数据来源", status=422, code="regularization.sources")
        for path in sorted(iter_images(root)):
            if not path.resolve().is_relative_to(root):
                raise ApiError("训练图片链接超出所选目录", status=403, code="regularization.path")
            relative = path.relative_to(root).as_posix()
            counts["source_images"] += 1
            digest = content_hash(path)
            identity = _digest([str(root), relative, digest])
            provenance = {
                "identity": identity,
                "source_id": source["id"],
                "source_path": str(root),
                "rel_path": relative,
                "image_hash": digest,
            }
            caption = caption_target(path, source["caption_ext"], directory_cache=caption_directories)
            try:
                text = read_editable_caption(caption)
                tags = list(dict.fromkeys(tag.strip() for tag in text.split(",") if tag.strip()))
            except (OSError, UnicodeError, ValueError):
                counts["invalid_captions"] += 1
                signature_rows.append(
                    [identity, "invalid", content_hash(caption) if caption.is_file() else None]
                )
                continue
            signature_rows.append([identity, tags, identity in existing])
            if not tags:
                counts["missing_captions"] += 1
                continue
            per_image = {}
            for tag in tags:
                per_image.setdefault(normalize_tag(tag), tag)
            for key, tag in per_image.items():
                frequencies.setdefault(key, {"tag": tag, "count": 0})["count"] += 1
            prompt = ", ".join(tag for tag in tags if normalize_tag(tag) not in excluded)
            if not prompt:
                counts["empty_after_exclusion"] += 1
                continue
            if identity in existing:
                counts["existing_images"] += 1
                if request.generation_scope == "incremental":
                    continue
            candidates.append({"prompt": prompt, "training_source": provenance})
    selected = candidates[: request.count]
    plan = {
        "signature": _digest(
            [pid, vid, request.model_dump(mode="json", exclude={"plan_signature"}), signature_rows]
        ),
        "sources": [{key: source[key] for key in ("id", "path", "name")} for source in sources],
        "top_tags": sorted(frequencies.values(), key=lambda row: (-row["count"], row["tag"].casefold()))[
            :200
        ],
        **counts,
        "eligible_images": len(candidates),
        "planned_images": len(selected),
        "remaining_images": len(candidates) - len(selected),
        "max_batch_images": 200,
        "examples": [
            {
                "source_id": item["training_source"]["source_id"],
                "rel_path": item["training_source"]["rel_path"],
                "prompt": item["prompt"],
            }
            for item in selected[:8]
        ],
    }
    return plan, selected
