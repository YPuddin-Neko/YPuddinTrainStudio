"""Pinned MOSS-Image Base assets on the image model download queue."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import urllib.error
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .db import new_id, now
from .errors import ApiError, Conflict
from .flux3_downloads import BundleFile, _inside, bundle_source_url, cleanup_stage

MossBundleId = Literal["moss-image-base", "moss-image-text", "moss-image-vae"]
REPO = "OpenMOSS-Team/MOSS-Image-Base"
REVISION = "5f40f07a304541840b6019d2d34d30f966756787"
MS_REPO = "openmoss/MOSS-Image-Base"
MS_REVISION = "bf675045be106b8c620f856d555d218f881120a2"


class MossBundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: MossBundleId
    repo_id: Literal["OpenMOSS-Team/MOSS-Image-Base", "openmoss/MOSS-Image-Base"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: list[BundleFile] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def safe_files(self):
        seen = set()
        for item in self.files:
            parts = item.path.split("/")
            if (not item.sha256 or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
                or any(char in item.path for char in '\\:<>"|?*') or any(ord(char) < 32 for char in item.path)
                or PurePosixPath(item.path).is_absolute() or item.path.casefold() in seen or item.path.endswith(".py")):
                raise ValueError("invalid MOSS-Image bundle member")
            seen.add(item.path.casefold())
        return self

    @property
    def size(self):
        return sum(item.size for item in self.files)

    @property
    def digest(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# Both fixed official providers publish these exact runtime bytes. Remote Python is excluded.
_FILES = (
    ('LICENSE', 11358, 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'),
    ('model_index.json', 412, 'ca80b1f562a506e0eb24afb1acb464fb583163e3c240c04e4bc69fea090db173'),
    ('scheduler/scheduler_config.json', 494, '707322832cc507dfb8c51c0e0cd04d44ee2d30a9b036b6f3c47d6c17aa71adf0'),
    ('text_encoder/config.json', 2813, 'b31ee18a242bcfcccbe740f825f6b1d3ae231b0655eeecb5de3e98d504b7d68c'),
    ('text_encoder/model-00001-of-00004.safetensors', 4973910744, 'abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b'),
    ('text_encoder/model-00002-of-00004.safetensors', 4988364288, '9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894'),
    ('text_encoder/model-00003-of-00004.safetensors', 4954202112, '956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc'),
    ('text_encoder/model-00004-of-00004.safetensors', 1869001848, '9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208'),
    ('text_encoder/model.safetensors.index.json', 64641, '0af6fa3fa36ba8dcbb6b7ad0325a36c5b1b69862a9a1f1486b1b80385bd8f5ee'),
    ('tokenizer/chat_template.jinja', 7756, 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'),
    ('tokenizer/tokenizer.json', 19989325, '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'),
    ('tokenizer/tokenizer_config.json', 1123, 'bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87'),
    ('transformer/config.json', 240, '3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666'),
    ('transformer/diffusion_pytorch_model-00001-of-00003.safetensors', 4991978016, 'ae085fba123b6df67daf90aa9698fadee60c73e45bcaf3f3043d15f34e7a1804'),
    ('transformer/diffusion_pytorch_model-00002-of-00003.safetensors', 4983137016, '806e876d80c7fba560663a3a95580e6c7033da3f9ea3c52765245ef0e1818f75'),
    ('transformer/diffusion_pytorch_model-00003-of-00003.safetensors', 189549184, '5f319b72c40324294007c270d78112425627b537fcd8769b3c00f9f070e506a9'),
    ('transformer/diffusion_pytorch_model.safetensors.index.json', 40298, 'b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03'),
    ('vae/config.json', 838, 'cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1'),
    ('vae/diffusion_pytorch_model.safetensors', 249521372, 'c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a'),
)


def bundle_manifest(id_: MossBundleId, provider="huggingface"):
    prefixes = {"moss-image-base": ("",), "moss-image-text": ("text_encoder/", "tokenizer/"), "moss-image-vae": ("vae/",)}[id_]
    return MossBundleManifest(id=id_, repo_id=MS_REPO if provider == "modelscope" else REPO,
                              revision=MS_REVISION if provider == "modelscope" else REVISION,
                              files=[BundleFile(path=name, size=size, sha256=sha) for name, size, sha in _FILES
                                     if name == "LICENSE" or name.startswith(prefixes)])


def validate_request(body):
    if body.bundle not in {"moss-image-base", "moss-image-text", "moss-image-vae"}:
        raise ValueError("请选择对应的 MOSS-Image 目录包。")
    spec = bundle_manifest(body.bundle, body.provider)
    kind = {"moss-image-base": "dit", "moss-image-text": "text_encoder", "moss-image-vae": "vae"}[spec.id]
    if (body.family != "moss_image" or body.kind != kind or body.mirror != "official" or body.url
        or body.repo_id not in (None, spec.repo_id) or body.revision not in (None, spec.revision)
        or body.filename not in (None, "", "text_encoder", "vae") or body.variant is not None
        or body.purpose != "training" or body.dtype not in (None, "fp32" if kind == "vae" else "bf16")
        or body.layout_version != 2):
        raise ValueError("MOSS-Image 目录包仅使用固定官方仓库、版本和文件清单。")
    return spec


def _group(storage, role):
    from .model_layout import shared_component_folder
    from .moss_image_components import component_identity

    return shared_component_folder(storage, role) / ("moss-" + component_identity(role)[:16])


def _roles(spec):
    return {"moss-image-base": ("text_encoder", "vae"), "moss-image-text": ("text_encoder",), "moss-image-vae": ("vae",)}[spec.id]


def _group_files(spec, role):
    prefixes = ("text_encoder/", "tokenizer/") if role == "text_encoder" else ("vae/",)
    return [item for item in spec.files if item.path == "LICENSE" or item.path.startswith(prefixes)]


def available_bundle(context, path, spec):
    from .moss_image_components import manifest_identity, proof_key, resolve_components, valid_component

    if spec.id != "moss-image-base":
        role = _roles(spec)[0]
        proof = context.db.get_kv(proof_key(path), {})
        return valid_component(context, proof, role)
    proof = context.db.get_kv(proof_key(path), {})
    return (proof.get("manifest_digest") == spec.digest or proof.get("official_identity") == manifest_identity(spec)) and resolve_components(context, path) is not None


def _recover_shared(service, row, spec):
    from .flux3_components import _files, _safe
    from .moss_image_components import MARKER, authenticate, component_paths, proof_key, valid_component

    storage = Path(row["storage_root"])
    for role in _roles(spec):
        target = _inside(storage, _group(storage, role))
        components = component_paths(target, role)
        proof = service.context.db.get_kv(proof_key(components[role]), {})
        if valid_component(service.context, proof, role):
            continue
        marker = target / MARKER
        if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 1024 * 1024:
            continue
        saved = json.loads(marker.read_text())
        expected = {item.path: item for item in _group_files(spec, role)}
        if (saved.get("download_id") != row["id"] or saved.get("manifest_digest") != spec.digest
            or saved.get("files") != {name: item.sha256 for name, item in expected.items()}
            or set(_files(service.context, _safe(service.context, target))) != set(expected) | {MARKER}):
            continue
        license_path = target / "LICENSE"
        if hashlib.sha256(license_path.read_bytes()).hexdigest() != expected["LICENSE"].sha256:
            continue
        authenticate(service.context, components, role, owner=row["id"])


def recover_publication(service, row):
    from .flux3_components import _files, registered_under, signature
    from .model_layout import model_folder
    from .moss_image_components import MARKER, component_paths, resolve_components

    spec = MossBundleManifest.model_validate(row["manifest"])
    _recover_shared(service, row, spec)
    storage, target = Path(row["storage_root"]), Path(row["target_path"])
    if spec.id != "moss-image-base":
        role = _roles(spec)[0]
        if target != _group(storage, role):
            return
        components = component_paths(target, role)
        primary = components[role]
        valid = available_bundle(service.context, Path(primary), spec)
    else:
        if target != model_folder(storage, "moss_image", "dit") / spec.digest[:12]:
            return
        primary = str(target)
        components = resolve_components(service.context, target)
        valid = available_bundle(service.context, target, spec)
    if valid:
        from .routes_core import register_moss_components

        with service.context.db.transaction():
            register_moss_components(target, service.context, row["is_default"], kind=row["kind"])
        asset = service.context.db.fetchone("SELECT id FROM models WHERE family='moss_image' AND kind=? AND path=?",
                                           (row["kind"], primary))
        if asset:
            row.update(status="completed", model_id=asset["id"], components=components, error=None, bytes_per_second=0,
                       downloaded_bytes=row["total_bytes"], eta_seconds=0, finished_at=now())
            return
    if spec.id != "moss-image-base":
        return
    marker = target / MARKER
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 1024 * 1024:
        return
    saved = json.loads(marker.read_text())
    if saved.get("download_id") != row["id"] or saved.get("manifest_digest") != spec.digest:
        return
    files = _files(service.context, target)
    if (set(files) == set(saved.get("owned_files", {})) | {MARKER}
        and all(signature(files[name]) == item["stat"] for name, item in saved["owned_files"].items())
        and not registered_under(service.context, target)):
        shutil.rmtree(_inside(storage, target))


def start_bundle(service, body, recommendation=None, *, frozen=None, layout_version=None, plan=None):
    from .model_downloads import ACTIVE, ModelDownload
    from .model_layout import model_folder
    from .model_recommendations import find_recommendation, related_download
    from .moss_image_components import find_components

    with service._starting:
        spec = MossBundleManifest.model_validate(frozen) if frozen else validate_request(body)
        if frozen and spec != validate_request(body):
            raise ValueError("MOSS-Image 下载清单与固定版本不符。")
        entry = recommendation or find_recommendation(spec.id)
        if entry and (active := related_download(entry, service.list())):
            if (active.get("recommendation_id") or active.get("bundle")) == entry.id:
                raise Conflict("this model file is already downloading", code="download.active", details={"download_id": active["id"]})
            return dict(active)
        storage = Path(service.context.settings()["paths"]["models_dir"]).resolve()
        if not service.context.is_allowed(storage):
            raise ApiError("model directory is outside allowed storage roots", status=403)
        for previous in service.list():
            if previous.get("family") == "moss_image" and previous["status"] in {"failed", "cancelled"}:
                recover_publication(service, previous)
                if previous["status"] == "completed":
                    service._update(previous["id"], **{key: previous[key] for key in (
                        "status", "model_id", "components", "error", "bytes_per_second", "downloaded_bytes", "eta_seconds", "finished_at")})
                    if previous["bundle"] == spec.id:
                        return dict(service.tasks[previous["id"]])
        components = find_components(service.context, candidates=True)
        target = (model_folder(storage, "moss_image", "dit") / spec.digest[:12] if body.kind == "dit"
                  else _group(storage, body.kind))
        _inside(storage, target)
        if body.kind == "dit" and target.exists():
            raise Conflict("this model directory already exists", code="download.exists")
        files = [item for item in spec.files if not item.path.startswith(("text_encoder/", "tokenizer/", "vae/"))] if body.kind == "dit" else []
        missing = [role for role in _roles(spec) if role not in components]
        total = sum(item.size for item in files) + sum(item.size for role in missing for item in _group_files(spec, role))
        with service.lock:
            if service.closed:
                raise Conflict("download service is stopping")
            if any(row.get("family") == "moss_image" and row["status"] in ACTIVE
                   and set(_roles(MossBundleManifest.model_validate(row["manifest"]))) & set(_roles(spec))
                   for row in service.tasks.values()):
                raise Conflict("a shared MOSS-Image component is downloading", code="download.active")
            id_ = new_id("dl")
            row = ModelDownload(id=id_, family="moss_image", kind=body.kind, provider=body.provider,
                                source_url=bundle_source_url(spec, body.provider), filename=spec.id,
                                target_path=str(target), status="queued", created_at=now(),
                                dtype="fp32" if body.kind == "vae" else "bf16", is_default=body.is_default,
                                bundle=spec.id, manifest=spec, layout_version=2, expected_size=total, total_bytes=total,
                                recommendation_id=entry.id if entry else None,
                                verification_source="pinned_revision_and_sha256").model_dump()
            row.update(storage_root=str(storage), component_plan={"components": components})
            service.tasks[id_], service.cancelled[id_] = row, threading.Event()
            service.pool.submit(service._run, id_, storage)
            started = dict(row)
        service._persist()
        return started


def run_bundle(service, id_, storage):
    from .download_errors import download_auth_error, download_http_error
    from .flux3_components import _json, _record, registered_under
    from .model_downloads import BundleTransfer, _Cancelled
    from .moss_image_components import (
        DESCRIPTION,
        MARKER,
        authenticate,
        certify_transferred,
        component_paths,
        proof_key,
        reference_records,
        valid_component,
        valid_snapshot,
        validate_component,
    )
    from .routes_core import ModelBody, add_model, register_moss_components

    row, context = service.tasks[id_], service.context
    spec = MossBundleManifest.model_validate(row["manifest"])
    stage, target = storage / ".downloads" / id_, Path(row["target_path"])
    transfer = BundleTransfer(service, id_, spec)
    published, registered = False, False
    try:
        transfer.cancelled()
        _inside(storage, stage).mkdir(parents=True, exist_ok=False)
        service._update(id_, status="downloading")
        components = dict(row["component_plan"]["components"])
        for role in _roles(spec):
            if role in components:
                proof = context.db.get_kv(proof_key(components[role]), {})
                if not valid_component(context, proof, role):
                    authenticate(context, components, role, transfer.event)
                continue
            shared = _inside(storage, _group(storage, role))
            paths = component_paths(shared, role)
            proof = context.db.get_kv(proof_key(paths[role]), {})
            if shared.exists():
                if not valid_component(context, proof, role):
                    raise ValueError("MOSS-Image 共享目录已存在，但缺少有效校验记录。")
                row["total_bytes"] -= sum(item.size for item in _group_files(spec, role))
                service._update(id_, total_bytes=row["total_bytes"], expected_size=row["total_bytes"])
            else:
                temporary = stage / role
                temporary.mkdir()
                digests = {item.path: transfer.transfer(item, _inside(storage, temporary / item.path))
                           for item in _group_files(spec, role)}
                validate_component(role, component_paths(temporary, role))
                _json(temporary / MARKER, {"download_id": id_, "manifest_digest": spec.digest, "files": digests})
                with service.lock:
                    transfer.cancelled()
                    shared.parent.mkdir(parents=True, exist_ok=True)
                    if shared.exists():
                        raise ValueError("MOSS-Image 共享目录在下载期间发生变化。")
                    temporary.rename(shared)
                certify_transferred(context, paths, role, owner=id_)
            components.update(paths)
        references = reference_records(context, {key: value for key, value in components.items()
                                                if key in {"tokenizer", *_roles(spec)}})
        if row["kind"] == "dit":
            own = stage / "installation"
            own.mkdir()
            digests = {item.path: transfer.transfer(item, _inside(storage, own / item.path)) for item in spec.files
                       if not item.path.startswith(("text_encoder/", "tokenizer/", "vae/"))}
            from ypuddin.models.moss_image.loading import transformer_config

            transformer_config(own / "transformer")
            digests[DESCRIPTION] = _json(own / DESCRIPTION, {"format": "ypuddin-moss-image-components", "version": 1,
                                                          "components": components})
            receipt = {"download_id": id_, "layout_version": 2, "manifest_digest": spec.digest,
                       "components": components, "references": references,
                       "owned_files": _record(context, own, digests)["files"]}
            digests[MARKER] = _json(own / MARKER, receipt)
            if any(not valid_snapshot(context, record) for record in references.values()):
                raise ValueError("MOSS-Image 共享组件在下载期间发生变化。")
            with service.lock:
                transfer.cancelled()
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError("MOSS-Image 安装目录已存在。")
                own.rename(target)
                published = True
            with context.db.transaction():
                context.db.set_kv(proof_key(target), {**receipt, "owned": _record(context, target, digests)})
                asset = add_model(ModelBody(family="moss_image", kind="dit", path=str(target), dtype="bf16",
                                            is_default=row["is_default"]), context)
        else:
            transfer.cancelled()
            with context.db.transaction():
                assets = register_moss_components(target, context, row["is_default"], kind=row["kind"], components=components)
                asset = next(item for item in assets if item["kind"] == row["kind"])
        registered = True
        service._update(id_, status="completed", model_id=asset["id"], components=components,
                        downloaded_bytes=transfer.done, bytes_per_second=0, eta_seconds=0, finished_at=now())
    except _Cancelled:
        service._update(id_, status="cancelled", finished_at=now())
    except Exception as error:
        if published and not registered:
            with context.db.transaction():
                if not registered_under(context, target):
                    shutil.rmtree(_inside(storage, target))
        message = transfer.policy.redact(error)
        if transfer.token:
            message = message.replace(transfer.token, "[redacted]")
        if isinstance(error, urllib.error.HTTPError):
            message = download_auth_error(error, provider=row["provider"], authenticated=bool(transfer.token), mirror=False) or download_http_error(error.code)
        service._update(id_, status="cancelled" if transfer.event.is_set() else "failed",
                        error=None if transfer.event.is_set() else message, finished_at=now())
    finally:
        try:
            cleanup_stage(storage, id_)
        except (OSError, ValueError):
            pass
