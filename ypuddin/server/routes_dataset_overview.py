"""Read-only overview statistics from the complete, version-scoped dataset index."""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from . import models as m
from .context import ServiceContext
from .dataset_sort import sort_images
from .errors import ApiError, NotFound
from .routes_work import _dataset_image_items, _get_dataset, ctx

router = APIRouter()


class FolderCount(BaseModel):
    path: str
    count: int


class DatasetOverview(BaseModel):
    dataset_id: str
    folders: list[FolderCount]
    stats: m.DatasetStats
    caption_stats: m.DatasetCaptionStats
    images: m.ImagePage


@router.get("/datasets/{did}/overview", response_model=DatasetOverview)
def dataset_overview(
    did: str,
    project_id: str,
    version_id: str | None = None,
    folder: str = "",
    q: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(16, ge=1, le=100),
    c: ServiceContext = Depends(ctx),
    sort: Literal["filename", "folder", "modified"] = "filename",
) -> dict:
    row = _get_dataset(c, did)
    if row["project_id"] != project_id or row.get("version_id") != version_id:
        raise NotFound("Dataset does not belong to this project version", code="dataset.not_found")
    if row.get("index_status") != "ready":
        raise ApiError("Dataset index is not ready", code="dataset.index_not_ready", status=409)
    if folder and (
        PurePosixPath(folder).is_absolute() or ".." in PurePosixPath(folder).parts or "\\" in folder
    ):
        raise ApiError("Invalid relative folder", code="dataset.folder", status=422)

    items = [
        {**item, "rel_path": item["rel_path"].replace("\\", "/")} for item in _dataset_image_items(c, row)
    ]
    folder_counts: Counter[str] = Counter()
    for item in items:
        parts = PurePosixPath(item["rel_path"]).parts[:-1]
        for depth in range(1, len(parts) + 1):
            folder_counts["/".join(parts[:depth])] += 1
    selected = [item for item in items if not folder or item["rel_path"].startswith(folder.rstrip("/") + "/")]
    resolutions: Counter[tuple[int, int]] = Counter()
    ratios: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    tags: dict[str, dict] = {}
    counts = {"images": len(selected), "captioned": 0, "missing": 0, "invalid": 0}
    for item in selected:
        counts[item["caption_status"]] += 1
        if item["caption_format"]:
            formats[item["caption_format"]] += 1
        if item["width"] > 0 and item["height"] > 0:
            resolutions[(item["width"], item["height"])] += 1
            ratios[str(round(item["width"] / item["height"], 1))] += 1
        for token in item["_tokens"]:
            entry = tags.setdefault(token.casefold(), {"tag": token, "count": 0})
            entry["count"] += 1
    matching = [
        item
        for item in selected
        if not q.strip()
        or q.strip().casefold() in item["caption"].casefold()
        or q.strip().casefold() in item["rel_path"].casefold()
    ]
    matching = sort_images(matching, row["path"], sort)
    start = (page - 1) * page_size
    return {
        "dataset_id": did,
        "folders": [{"path": name, "count": count} for name, count in sorted(folder_counts.items())],
        "stats": {
            "images": len(selected),
            "captioned": counts["captioned"],
            "masks": sum(bool(item["has_mask"]) for item in selected),
            "resolutions": [
                {"w": w, "h": h, "count": count}
                for (w, h), count in sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))
            ],
            "ar_hist": [
                {"ar": ratio, "count": count}
                for ratio, count in sorted(ratios.items(), key=lambda item: float(item[0]))
            ],
        },
        "caption_stats": {
            **counts,
            "formats": dict(formats),
            "unique_tags": len(tags),
            "tags": sorted(tags.values(), key=lambda item: (-item["count"], item["tag"].casefold())),
        },
        "images": {
            "items": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in matching[start : start + page_size]
            ],
            "total": len(matching),
            "page": page,
            "page_size": page_size,
        },
    }
