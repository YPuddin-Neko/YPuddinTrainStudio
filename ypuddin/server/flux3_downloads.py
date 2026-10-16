"""Fixed FLUX.3 directory bundles on the image model download queue."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .db import new_id, now
from .errors import ApiError, Conflict
from .model_credentials import Provider

BundleId = Literal["flux3-base", "flux3-text"]
REPO = "black-forest-labs/flux-3-image-klein-base"
REVISION = "17333191c36bae1b61b0a247e33ba284f1df07fa"
MARKER = ".ypuddin-flux3-download.json"


class BundleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    size: int = Field(gt=0)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    git_blob_sha1: str | None = Field(None, pattern=r"^[0-9a-f]{40}$")


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: BundleId
    repo_id: Literal["black-forest-labs/flux-3-image-klein-base"] = REPO
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: list[BundleFile] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def safe_files(self):
        seen = set()
        for item in self.files:
            parts = item.path.split("/")
            if (not item.path or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
                or any(char in item.path for char in '\\:<>"|?*') or any(ord(char) < 32 for char in item.path)
                or PurePosixPath(item.path).is_absolute() or item.path.casefold() in seen
                or self.id == "flux3-text" and not item.path.startswith("text_encoder/")):
                raise ValueError("invalid FLUX.3 bundle file path")
            seen.add(item.path.casefold())
        return self

    @property
    def size(self):
        return sum(item.size for item in self.files)

    @property
    def digest(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def destination(self, item):
        return item.path.removeprefix("text_encoder/") if self.id == "flux3-text" else item.path


# Public HF tree at REVISION. Gated LFS digests were withheld; no SHA-256 is inferred from Git OIDs.
_FILES = (
    ('LICENSE.md', 19593, '044cf48568c278150d132a7479ea237a79766ea7'),
    ('NOTICE.md', 330, 'ee97a99bbf8ad09b197cf3cb6a079084876b0376'),
    ('ae.safetensors', 336211292, None),
    ('config.json', 805, '88305de45c2a050899303d2a63dc8fa9a710ab7c'),
    ('manifest.json', 404, '5d8344c8e4a6125c5a4de05523c2611149a4e42e'),
    ('model.safetensors', 11076957072, None),
    ('text_encoder/LICENSE', 11357, '261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64'),
    ('text_encoder/README.md', 7133, '542417095b66d3fc13be22bbf6722928787b8ef4'),
    ('text_encoder/chat_template.json', 5502, 'e49e75bdda104c8de941eb20e0b3765184fb27bc'),
    ('text_encoder/config.json', 1505, '22be8bb79043d39840ca5e70d7a2dab80311602c'),
    ('text_encoder/generation_config.json', 269, 'e116347d7ef42621c7a5957f82c015c38d939c1f'),
    ('text_encoder/merges.txt', 1671839, '20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0'),
    ('text_encoder/model-00001-of-00002.safetensors', 4967229296, None),
    ('text_encoder/model-00002-of-00002.safetensors', 3908490048, None),
    ('text_encoder/model.safetensors.index.json', 64742, '3b52de97496e2f4501dfa92a87642f60db3e22a1'),
    ('text_encoder/preprocessor_config.json', 390, '2ea84a437d448ff71b08df68fdd949d5cc4ebb64'),
    ('text_encoder/tokenizer.json', 7032403, 'c6cc1014128b19d1fc46b1d30a23e3b1d35db421'),
    ('text_encoder/tokenizer_config.json', 10868, 'd3d3763207692c78780f4bf42d4dadf49a5c8012'),
    ('text_encoder/video_preprocessor_config.json', 385, '3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe'),
    ('text_encoder/vocab.json', 2776833, '4783fe10ac3adce15ac8f358ef5462739852c569'),
)


MODELSCOPE_REVISION = '018fa78bb38c828d8844360b1e1952abcf88875a'
# ModelScope repo/files at MODELSCOPE_REVISION supplies SHA-256 for every bundle member.
MODELSCOPE_SHA256 = {
    'LICENSE.md': '100a3a5af8f1a2943dca0d127671bcaa93ad894a63330825219f1a0a44451f22',
    'NOTICE.md': 'a12f7eb0cab87ed210ffb0921425589a1c88abded6ba0c4285b4d32cbaf5f35c',
    'ae.safetensors': '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3',
    'config.json': '89cc799ddb8c0f5a60104d9312824c37cecb923e6a08dfe30a70e4fa2e850535',
    'manifest.json': 'f79410b5b24d0a9a41dd7038eaa9af39353794bc2e7ee9cdcc5859826280377b',
    'model.safetensors': '5c921759dc03663dbb67df11d8e3d317d2d5b893b31b969c95dc77ebf861b19c',
    'text_encoder/LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'text_encoder/README.md': 'a884e5e78f7d6f7bfe237f909dbc41a126542e259dc79d8ab33cc8980580ff79',
    'text_encoder/chat_template.json': '6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4',
    'text_encoder/config.json': 'edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d',
    'text_encoder/generation_config.json': '8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44',
    'text_encoder/merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'text_encoder/model-00001-of-00002.safetensors': '30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9',
    'text_encoder/model-00002-of-00002.safetensors': '046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6',
    'text_encoder/model.safetensors.index.json': '58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b',
    'text_encoder/preprocessor_config.json': '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'text_encoder/tokenizer.json': 'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7',
    'text_encoder/tokenizer_config.json': 'c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399',
    'text_encoder/video_preprocessor_config.json': '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
    'text_encoder/vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}


def bundle_manifest(id_: BundleId, provider: Provider = "huggingface") -> BundleManifest:
    if provider == "modelscope":
        return BundleManifest(id=id_, revision=MODELSCOPE_REVISION, files=[
            BundleFile(path=path, size=size, sha256=MODELSCOPE_SHA256[path]) for path, size, _ in _FILES
            if id_ == "flux3-base" or path.startswith("text_encoder/")
        ])
    return BundleManifest(id=id_, revision=REVISION, files=[
        BundleFile(path=path, size=size, git_blob_sha1=oid) for path, size, oid in _FILES
        if id_ == "flux3-base" or path.startswith("text_encoder/")
    ])


def validate_request(body):
    spec = bundle_manifest(body.bundle, body.provider)
    if (body.family != "flux3" or body.kind != ("dit" if body.bundle == "flux3-base" else "text_encoder")
        or body.mirror != "official" or body.url
        or body.repo_id not in (None, REPO) or body.revision not in (None, spec.revision)
        or body.filename not in (None, "", "text_encoder") or body.variant is not None
        or body.purpose != "training" or body.dtype not in (None, "bf16")):
        raise ValueError("FLUX.3 目录包仅使用固定官方仓库、版本和文件清单。")
    return spec


def bundle_source_url(spec, provider, filename=None):
    if provider == "modelscope":
        query = {"Revision": spec.revision}
        if filename is not None:
            query["FilePath"] = filename
        return f"https://modelscope.cn/api/v1/models/{spec.repo_id}/repo?{urllib.parse.urlencode(query)}"
    path = urllib.parse.quote(filename, safe="/") if filename is not None else ""
    return f"https://huggingface.co/{spec.repo_id}/resolve/{spec.revision}/{path}"


def _inside(root, path):
    if root.resolve() != root or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("FLUX.3 download path escapes model storage")
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("FLUX.3 download path contains a symlink")
    return path


def cleanup_stage(root, id_):
    if not re.fullmatch(r"dl_[0-9a-f]{12}", id_):
        return
    stage = _inside(root, root / ".downloads" / id_)
    if stage.exists():
        shutil.rmtree(stage)


def _proof_key(path):
    return "model.flux3_bundle." + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def _stat(path):
    value = path.stat()
    return [value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_dev]


def _exact_members(path, spec):
    files = {spec.destination(item) for item in spec.files} | {MARKER}
    directories = {str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent) != "."}
    for relative in {"", *directories}:
        directory = path / relative
        if directory.is_symlink() or not directory.is_dir():
            return False
        expected = {name.rsplit("/", 1)[-1] for name in files | directories
                    if str(PurePosixPath(name).parent) == (relative or ".")}
        members = list(directory.iterdir())
        if {member.name for member in members} != expected or any(member.is_symlink() for member in members):
            return False
    return all((path / name).is_file() for name in files)


def available_bundle(context, path, spec):
    """Reuse the completed transfer proof while every member retains its filesystem identity."""
    try:
        if not context.is_allowed(path.resolve()) or path.is_symlink() or not _exact_members(path, spec):
            return False
        proof = context.db.get_kv(_proof_key(path), {})
        if proof.get("manifest_digest") != spec.digest:
            return False
        for item in spec.files:
            local = _inside(path, path / spec.destination(item))
            if _stat(local) != proof["files"][item.path]["stat"]:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def recover_publication(service, row):
    """Reconcile only an interrupted task's unchanged, explicitly owned publication."""
    from .model_layout import model_folder

    root, target = Path(row["storage_root"]), Path(row["target_path"])
    spec = BundleManifest.model_validate(row["manifest"])
    if (not service.context.is_allowed(root) or
        target != model_folder(root, "flux3", row["kind"]) / spec.digest[:12]):
        return
    _inside(root, target)
    if not _exact_members(target, spec):
        return
    marker = target / MARKER
    if marker.stat().st_size > 1024 * 1024:
        return
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    if (not isinstance(receipt, dict) or receipt.get("download_id") != row["id"] or receipt.get("manifest_digest") != spec.digest
        or receipt.get("manifest") != spec.model_dump()):
        return
    assets = service.context.db.fetchall("SELECT * FROM models WHERE path=?", (str(target),))
    if assets:
        asset = next((asset for asset in assets if asset["family"] == "flux3" and asset["kind"] == row["kind"]), None)
        if asset and available_bundle(service.context, target, spec):
            row.update(status="completed", error=None, model_id=asset["id"], downloaded_bytes=spec.size,
                       bytes_per_second=0, eta_seconds=0, finished_at=now())
        return
    if all(_stat(target / spec.destination(item)) == receipt["files"][item.path]["stat"] for item in spec.files):
        # No user changes and no registry entry: only this task's unpublished directory is removed.
        shutil.rmtree(_inside(root, target))


def validate_bundle(path, spec):
    """Check the complete component headers before publishing a directory bundle."""
    from ypuddin.models.flux3.latent import Flux3Latent
    from ypuddin.models.flux3.loading import transformer_config
    from ypuddin.models.flux3.text import Flux3Text

    if spec.id == "flux3-base":
        transformer_config(path / "model.safetensors")
        Flux3Latent.validate_path(path / "ae.safetensors")
        Flux3Text.validate_paths(path / "text_encoder")
    else:
        Flux3Text.validate_paths(path)


def start_bundle(service, body, recommendation=None, *, frozen=None):
    from .model_downloads import ACTIVE, ModelDownload
    from .model_layout import model_folder

    spec = BundleManifest.model_validate(frozen) if frozen else validate_request(body)
    root = Path(service.context.settings()["paths"]["models_dir"]).resolve()
    if not service.context.is_allowed(root):
        raise ApiError("model directory is outside allowed storage roots", status=403)
    target = _inside(root, model_folder(root, "flux3", body.kind) / spec.digest[:12])
    with service.lock:
        if service.closed:
            raise Conflict("download service is stopping")
        if target.exists() or any(row["target_path"] == str(target) and row["status"] in ACTIVE for row in service.tasks.values()):
            raise Conflict("this model directory exists or is downloading", code="download.exists")
        id_ = new_id("dl")
        row = ModelDownload(
            id=id_, family="flux3", kind=body.kind, provider=body.provider, mirror="official",
            source_url=bundle_source_url(spec, body.provider),
            filename=spec.id, target_path=str(target), status="queued", created_at=now(),
            dtype="bf16", is_default=body.is_default, purpose="training",
            expected_size=spec.size, total_bytes=spec.size, recommendation_id=recommendation.id if recommendation else None,
            bundle=spec.id, manifest=spec,
            verification_source="pinned_revision_and_sha256" if all(item.sha256 for item in spec.files) else "pinned_revision_and_sizes",
        ).model_dump()
        row["storage_root"] = str(root)
        service.tasks[id_] = row
        service.cancelled[id_] = threading.Event()
        service.pool.submit(service._run, id_, root)
        started = dict(row)
    service._persist()
    return started


def run_bundle(service, id_, root):
    from .download_errors import download_auth_error, download_http_error
    from .model_downloads import _Cancelled, _check_download_header, _Redirect
    from .network import ProxyPolicy
    from .routes_core import ModelBody, _check_model_admission, add_model

    row, event = service.tasks[id_], service.cancelled[id_]
    spec = BundleManifest.model_validate(row["manifest"])
    stage, target = root / ".downloads" / id_, Path(row["target_path"])
    token, published, registered, policy = None, False, False, ProxyPolicy()
    try:
        if event.is_set():
            raise _Cancelled()
        _inside(root, stage).mkdir(parents=True, exist_ok=False)
        service._update(id_, status="downloading")
        policy = ProxyPolicy.from_context(service.context)
        token = service.credentials.token(row["provider"], policy=policy)
        headers = {"User-Agent": "YPuddinTrainStudio", "Accept-Encoding": "identity"}
        if token:
            headers["Cookie" if row["provider"] == "modelscope" else "Authorization"] = (
                f"m_session_id={token}" if row["provider"] == "modelscope" else f"Bearer {token}"
            )
        done, last, previous = 0, time.monotonic(), 0
        records = {}
        for item in spec.files:
            if event.is_set():
                raise _Cancelled()
            destination = _inside(root, stage / spec.destination(item))
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle_source_url(spec, row["provider"], item.path)
            policy = ProxyPolicy.from_context(service.context)
            opener = service.opener or policy.opener(_Redirect())
            received, digest = 0, hashlib.sha256()
            git_digest = hashlib.sha1(f"blob {item.size}\0".encode()) if item.git_blob_sha1 else None
            probe = bytearray() if item.path.endswith(".safetensors") else None
            with opener.open(urllib.request.Request(source, headers=headers), timeout=15) as response, destination.open("xb") as stream:
                if response.status != 200:
                    raise ValueError(f"unexpected download HTTP status {response.status}")
                if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) != item.size:
                    raise ValueError(f"FLUX.3 file size differs: {item.path}")
                while chunk := response.read(256 * 1024):
                    if event.is_set():
                        raise _Cancelled()
                    received += len(chunk)
                    if received > item.size:
                        raise ValueError(f"FLUX.3 file exceeds its pinned size: {item.path}")
                    if probe is not None:
                        from .model_inspection import HEADER_LIMIT

                        probe.extend(chunk[:HEADER_LIMIT + 8 - len(probe)])
                        if len(probe) >= 8:
                            length = int.from_bytes(probe[:8], "little")
                            if not 2 <= length <= HEADER_LIMIT:
                                raise ValueError("invalid safetensors header")
                            if len(probe) >= length + 8:
                                _check_download_header(probe[8:length + 8], "flux3")
                                probe = None
                    digest.update(chunk)
                    if git_digest is not None:
                        git_digest.update(chunk)
                    stream.write(chunk)
                    done += len(chunk)
                    current = time.monotonic()
                    if current - last >= 0.4:
                        rate = (done - previous) / (current - last)
                        service._update(id_, downloaded_bytes=done, bytes_per_second=rate,
                                        eta_seconds=(spec.size - done) / rate, progress_at=now())
                        last, previous = current, done
                if (received != item.size or item.sha256 and digest.hexdigest() != item.sha256
                    or item.git_blob_sha1 and git_digest.hexdigest() != item.git_blob_sha1):
                    raise ValueError(f"FLUX.3 file integrity check failed: {item.path}")
                stream.flush()
                os.fsync(stream.fileno())
            records[item.path] = {"size": received, "sha256": digest.hexdigest(), "stat": _stat(destination),
                                  "verification": "sha256" if item.sha256 else "git_blob_sha1" if item.git_blob_sha1 else "revision_size_download_digest"}
        _check_model_admission(stage, "flux3", row["kind"], service.context)
        validate_bundle(stage, spec)
        receipt = {"download_id": id_, "manifest": spec.model_dump(), "manifest_digest": spec.digest, "files": records}
        (stage / MARKER).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        with service.lock:
            if event.is_set() or service.closed:
                raise _Cancelled()
            _inside(root, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError("FLUX.3 destination already exists")
            stage.rename(target)
            published = True
        for item in spec.files:
            records[item.path]["stat"] = _stat(target / spec.destination(item))
        with service.context.db.transaction():
            asset = add_model(ModelBody(family="flux3", kind=row["kind"], path=str(target), dtype=row["dtype"], is_default=row["is_default"]), service.context)
            service.context.db.set_kv(_proof_key(target), {"manifest_digest": spec.digest, "files": records})
        registered = True
        service._update(id_, status="completed", model_id=asset["id"], downloaded_bytes=done,
                        bytes_per_second=0, eta_seconds=0, finished_at=now())
    except _Cancelled:
        service._update(id_, status="cancelled", finished_at=now())
    except Exception as error:
        if published and not registered:
            _inside(root, target)
            shutil.rmtree(target)
        message = policy.redact(error)
        if token:
            message = message.replace(token, "[redacted]")
        if isinstance(error, urllib.error.HTTPError):
            message = download_auth_error(error, provider=row["provider"], authenticated=bool(token), mirror=False) or download_http_error(error.code)
        service._update(id_, status="cancelled" if event.is_set() else "failed", error=None if event.is_set() else message, finished_at=now())
    finally:
        try:
            cleanup_stage(root, id_)
        except (OSError, ValueError):
            pass
