"""Dataset naming, repeat counts and reversible training membership."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from . import models as m
from .context import ServiceContext
from .dataset_uploads import relative_upload_path
from .errors import ApiError
from .source_roles import managed_source_role
from .versions import file_manifest

router = APIRouter()


def context(request: Request) -> ServiceContext:
    return request.app.state.ctx


def source_states(c: ServiceContext, row: dict) -> list[tuple]:
    from .routes_work import get_project_config

    if not row.get("project_id"):
        return [(Path(row["path"]).expanduser().resolve(), set(), (), row)]
    config = get_project_config(row["project_id"], c, row.get("version_id"))
    return [
        (
            Path(s["path"]).expanduser().resolve(),
            set(s.get("excluded_files", [])),
            tuple(d + "/" for d in s.get("excluded_dirs", [])),
            s,
        )
        for s in config.get("dataset", {}).get("sources", [])
        if s.get("path")
    ]


def included(path: str, states: list[tuple]) -> bool:
    image = Path(path)
    for root, excluded, directories, _ in states:
        if image.is_relative_to(root):
            relative = image.relative_to(root).as_posix()
            if relative not in excluded and not relative.startswith(directories):
                return True
    return False


def can_rename(c: ServiceContext, row: dict) -> bool:
    if not row.get("project_id"):
        return False
    path = Path(row["path"])
    role = managed_source_role(c, row["project_id"], row.get("version_id"), str(path))
    return bool(role and path.resolve() != Path(role[1]).resolve() and path.absolute() == path.resolve())


def _atomic_bytes(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class DatasetEdit(BaseModel):
    name: str | None = None
    repeats: int | None = Field(None, ge=1, le=1_000_000)
    masked_loss: bool | None = None


class MembershipEdit(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=20000)
    included: bool


class ImageReference(BaseModel):
    dataset_id: str
    rel_path: str


class VersionMembershipEdit(BaseModel):
    images: list[ImageReference] = Field(min_length=1, max_length=20000)
    included: bool


def _independent_source(config: dict, row: dict) -> dict:
    sources = config.setdefault("dataset", {}).setdefault("sources", [])
    root = Path(row["path"]).resolve()
    for source in sources:
        if Path(source["path"]).expanduser().resolve() == root:
            return source
    # Split a registered subfolder from an ancestor source without double-counting it.
    parents = [s for s in sources if root.is_relative_to(Path(s["path"]).expanduser().resolve())]
    if parents:
        original = max(parents, key=lambda s: len(Path(s["path"]).parts))
        prefix = root.relative_to(Path(original["path"]).expanduser().resolve()).as_posix()
        source = copy.deepcopy(original)
        source["excluded_files"] = [
            p[len(prefix) + 1 :] for p in original.get("excluded_files", []) if p.startswith(prefix + "/")
        ]
        source["excluded_dirs"] = [
            p[len(prefix) + 1 :] for p in original.get("excluded_dirs", []) if p.startswith(prefix + "/")
        ]
        if any(prefix == p or prefix.startswith(p + "/") for p in original.get("excluded_dirs", [])):
            from ypuddin.data.index import iter_images

            source["excluded_files"] = [p.relative_to(root).as_posix() for p in iter_images(root)]
        for parent in parents:
            relative = root.relative_to(Path(parent["path"]).expanduser().resolve()).as_posix()
            parent["excluded_dirs"] = sorted(set(parent.get("excluded_dirs", [])) | {relative})
    else:
        source = {k: row[k] for k in ("repeats", "caption_ext", "is_reg", "prior_weight", "class_prompt")}
        source["is_reg"] = bool(source["is_reg"])
    source["path"] = str(root)
    sources.append(source)
    return source


@router.post("/datasets/{did}/membership", response_model=m.MembershipResult)
def edit_membership(did: str, body: MembershipEdit, c: ServiceContext = Depends(context)) -> dict:
    from .routes_work import _get_dataset

    row = _get_dataset(c, did)
    return edit_version_membership(
        row["project_id"],
        row["version_id"],
        VersionMembershipEdit(
            images=[ImageReference(dataset_id=did, rel_path=p) for p in body.paths],
            included=body.included,
        ),
        c,
    )


@router.post("/projects/{pid}/versions/{vid}/dataset-membership", response_model=m.MembershipResult)
def edit_version_membership(
    pid: str, vid: str, body: VersionMembershipEdit, c: ServiceContext = Depends(context)
) -> dict:
    from ypuddin.data.index import iter_images

    from .routes_work import _get_dataset, _write_project_config, get_project_config

    grouped: dict[str, set[str]] = {}
    for image in body.images:
        grouped.setdefault(image.dataset_id, set()).add(relative_upload_path(image.rel_path).as_posix())
    with c.versions.mutation(pid, vid), c.db.lock:
        config = get_project_config(pid, c, vid)
        for did, paths in grouped.items():
            row = _get_dataset(c, did)
            if row["project_id"] != pid or row["version_id"] != vid:
                raise ApiError("所选图片不属于当前版本。", code="dataset.scope", status=422)
            root = Path(row["path"]).resolve()
            # Unreadable images must also be removable from training.
            known = {p.relative_to(root).as_posix() for p in iter_images(root)}
            for relative in paths:
                path = root / relative
                if (
                    relative not in known
                    or not path.is_file()
                    or path.is_symlink()
                    or not path.resolve().is_relative_to(root)
                ):
                    raise ApiError("所选图片已变化，请刷新后重试。", code="dataset.image_path", status=409)
            source = _independent_source(config, row)
            excluded = set(source.get("excluded_files", []))
            source["excluded_files"] = sorted(excluded - paths if body.included else excluded | paths)
        _write_project_config(c, pid, config, vid)
    for did in grouped:
        c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "membership"})
    return {"changed": sum(map(len, grouped.values())), "included": body.included}


@router.patch("/datasets/{did}", response_model=m.DatasetInfo, response_model_exclude_unset=True)
def edit_dataset(did: str, body: DatasetEdit, c: ServiceContext = Depends(context)) -> dict:
    from .routes_work import (
        _dataset_row,
        _get_dataset,
        _records_path,
        _write_project_config,
        get_project_config,
    )

    row = _get_dataset(c, did)
    with c.versions.mutation(row["project_id"], row["version_id"]), c.db.lock:
        row = _get_dataset(c, did)
        old = Path(row["path"]).resolve()
        new = old
        if body.name is not None and body.name != old.name:
            name = relative_upload_path(body.name)
            if len(name.parts) != 1 or not can_rename(c, row):
                raise ApiError("只能重命名当前版本内的训练文件夹。", code="dataset.rename_path", status=400)
            new = old.with_name(str(name))
            if any(p.name.casefold() == new.name.casefold() for p in old.parent.iterdir()):
                raise ApiError("这个文件夹名称已存在。", code="dataset.name_exists", status=409)
            file_manifest(old)  # Reject redirected/special files before changing anything.
        config = get_project_config(row["project_id"], c, row["version_id"])
        previous = copy.deepcopy(config)
        backups: dict[Path, bytes] = {}
        moved = written = False
        try:
            with c.db.transaction():
                if body.masked_loss is not None:
                    config.setdefault("dataset", {})["masked_loss"] = body.masked_loss
                if body.repeats is not None:
                    _independent_source(config, row)["repeats"] = body.repeats
                    c.db.update("datasets", did, {"repeats": body.repeats})
                if new != old:
                    for section in ("dataset", "validation"):
                        for source in config.get(section, {}).get("sources", []):
                            path = Path(source["path"]).expanduser().resolve()
                            if path.is_relative_to(old):
                                source["path"] = str(new / path.relative_to(old))
                            elif old.is_relative_to(path):
                                before, after = (
                                    old.relative_to(path).as_posix(),
                                    new.relative_to(path).as_posix(),
                                )
                                for key in ("excluded_files", "excluded_dirs"):
                                    if key in source:
                                        source[key] = [
                                            after + p[len(before) :]
                                            if p == before or p.startswith(before + "/")
                                            else p
                                            for p in source[key]
                                        ]
                    for dataset in c.db.fetchall(
                        "SELECT * FROM datasets WHERE version_id=?", (row["version_id"],)
                    ):
                        path = Path(dataset["path"]).resolve()
                        if not path.is_relative_to(old):
                            continue
                        c.db.update("datasets", dataset["id"], {"path": str(new / path.relative_to(old))})
                        index = _records_path(c, dataset["id"])
                        if index.exists():
                            backups[index] = index.read_bytes()
                            records = json.loads(backups[index])
                            for record in records:
                                for key in ("path", "caption_path", "mask_path"):
                                    if record.get(key) and Path(record[key]).is_relative_to(old):
                                        record[key] = str(new / Path(record[key]).relative_to(old))
                            _atomic_bytes(index, json.dumps(records, ensure_ascii=False).encode())
                    old.rename(new)
                    moved = True
                _write_project_config(c, row["project_id"], config, row["version_id"])
                written = True
        except BaseException:
            if moved:
                new.rename(old)
            for path, content in backups.items():
                _atomic_bytes(path, content)
            if written:
                _write_project_config(c, row["project_id"], previous, row["version_id"])
            raise
    c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "settings"})
    return _dataset_row(c, _get_dataset(c, did), include_cache=False)
