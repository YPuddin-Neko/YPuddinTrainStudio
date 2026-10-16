"""FLUX.3 component references and stat-validated installation proofs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DESCRIPTION = ".ypuddin-flux3-components.json"
INSTALLATION = ".ypuddin-flux3-installation.json"
SHARED = ".ypuddin-flux3-shared.json"
KREA_TEXT_SHA256 = "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34"
KREA_TEXT_SIZE = 8875719384
VAE_SHA256 = "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"


def proof_key(path):
    return "model.flux3_components." + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()


def signature(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_dev]


def _safe(context, path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or path.resolve() != path or not context.is_allowed(path):
        raise ValueError("FLUX.3 component is outside allowed storage or contains a symlink")
    return path


def _files(context, path):
    path = _safe(context, path)
    if path.is_file():
        return {"": path}
    if not path.is_dir():
        raise ValueError("FLUX.3 component is missing")
    files = {}
    for parent, directories, names in os.walk(path, followlinks=False):
        for name in [*directories, *names]:
            member = _safe(context, Path(parent) / name)
            if member.is_file():
                files[member.relative_to(path).as_posix()] = member
            elif not member.is_dir():
                raise ValueError("FLUX.3 component contains a non-regular file")
    return files


def valid_snapshot(context, record):
    try:
        files = _files(context, record["path"])
        return set(files) == set(record["files"]) and all(
            signature(path) == record["files"][name]["stat"] for name, path in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def component_sha256(context, path):
    """Return existing proof only; catalog reads never hash model payloads."""
    from .flux3_downloads import _proof_key
    from .model_recommendations import cached_sha256

    try:
        path = _safe(context, path)
        if not path.is_file():
            return None
        digest = cached_sha256(context, path)
        if digest:
            return digest
        for parent in path.parents:
            for key in (_proof_key(parent), proof_key(parent)):
                proof = context.db.get_kv(key, {})
                item = proof.get("files", {}).get(path.relative_to(parent).as_posix(), {})
                if item.get("stat") == signature(path):
                    return item.get("sha256")
        record = context.db.get_kv(proof_key(path), {})
        item = record.get("files", {}).get("", {})
        return item.get("sha256") if item.get("stat") == signature(path) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def certify_component(context, path, event=None):
    """Explicit installation/import work may read each unverified file once."""
    files = _files(context, path)
    record = {"path": str(Path(path).resolve()), "files": {}}
    for name, member in files.items():
        before = signature(member)
        digest = component_sha256(context, member)
        if digest is None:
            hasher = hashlib.sha256()
            with member.open("rb") as source:
                while block := source.read(1024 * 1024):
                    if event is not None and event.is_set():
                        from .model_downloads import _Cancelled

                        raise _Cancelled()
                    hasher.update(block)
            digest = hasher.hexdigest()
        if signature(member) != before:
            raise ValueError("FLUX.3 component changed during verification")
        record["files"][name] = {"stat": before, "sha256": digest, "size": before[0]}
    if set(_files(context, path)) != set(files):
        raise ValueError("FLUX.3 component members changed during verification")
    return record


def certify_text_component(context, path, event=None):
    """Authenticate a manually imported official HF directory during an explicit operation."""
    from ypuddin.models.flux3.text import Flux3Text

    from .flux3_downloads import bundle_manifest

    if not metadata_text_candidate(context, path):
        raise ValueError("FLUX.3 text directory differs from the fixed official component sizes")
    record = certify_component(context, path, event)
    if Path(path).is_file():
        if record["files"][""]["sha256"] != KREA_TEXT_SHA256:
            raise ValueError("Qwen3-VL BF16 single-file digest differs from the fixed component")
        return {**record, "component": "text_encoder", "official_identity": "qwen3vl4b-v1"}
    spec = bundle_manifest("flux3-text", "modelscope")
    expected = {item.path.removeprefix("text_encoder/"): item for item in spec.files}
    if set(record["files"]) - set(expected) - {SHARED, ".ypuddin-flux3-download.json"}:
        raise ValueError("FLUX.3 official text directory has unrecognized members")
    required = set(expected) - {"LICENSE", "README.md"}
    if any(record["files"].get(name, {}).get("sha256") != item.sha256 or
           record["files"].get(name, {}).get("size") != item.size
           for name, item in expected.items() if name in required or name in record["files"]):
        raise ValueError("FLUX.3 text directory differs from the fixed official component")
    Flux3Text.validate_paths(path, path)
    return {**record, "component": "text_encoder", "official_identity": "qwen3vl4b-v1"}


def metadata_text_candidate(context, path):
    """Cheap candidate gate for explicit preparation, never proof of matching payloads."""
    from .flux3_downloads import bundle_manifest

    try:
        path = _safe(context, path)
        if path.is_file():
            if path.stat().st_size != KREA_TEXT_SIZE:
                return False
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as weights:
                keys = weights.keys()
                return ("model.language_model.embed_tokens.weight" in keys
                        and weights.get_slice("model.language_model.embed_tokens.weight").get_shape() == [151936, 2560]
                        and all(weights.get_slice(key).get_dtype() == "BF16" for key in keys))
        files = _files(context, path)
        expected = {item.path.removeprefix("text_encoder/"): item
                    for item in bundle_manifest("flux3-text", "modelscope").files}
        required = set(expected) - {"LICENSE", "README.md"}
        return (required <= set(files) and not set(files) - set(expected) - {SHARED, ".ypuddin-flux3-download.json"}
                and all(files[name].stat().st_size == expected[name].size for name in set(files) & set(expected)))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def metadata_candidates(context):
    """Unverified registered HF/file candidates for an explicit asynchronous install only."""
    for row in context.db.fetchall("SELECT * FROM models WHERE kind='text_encoder' ORDER BY is_default DESC, created_at DESC"):
        path = Path(row["path"])
        if metadata_text_candidate(context, path):
            result = {"text_encoder": str(path.resolve())}
            processor = _component_metadata(context, path).get("processor_path") or (path if path.is_dir() else path.parent)
            if value := _processor(context, processor):
                result["tokenizer"] = value
            return result
    return {}


def _component_metadata(context, path):
    from .model_recommendations import component_metadata

    return component_metadata(context, path)


def _processor(context, path):
    from .model_inspection import check_flux3_processor

    try:
        path = _safe(context, path)
        check_flux3_processor(path, context.is_allowed)
        return str(path)
    except (OSError, ValueError):
        return None


def resolve_components(context, root):
    """Project authenticated references; no registration or payload hashing on reads."""
    from .flux3_downloads import BundleManifest, _proof_key, available_bundle

    try:
        root = _safe(context, root)
        proof = context.db.get_kv(proof_key(root), {})
        if proof.get("layout_version") == 2:
            if not valid_snapshot(context, proof["owned"]):
                return None
            components = proof["components"]
            if not components or set(components) - {"text_encoder", "vae", "tokenizer"}:
                return None
            if any(proof["references"][role]["path"] != path or
                   not valid_snapshot(context, proof["references"][role]) for role, path in components.items()):
                return None
            if "vae" in components:
                description = json.loads((root / DESCRIPTION).read_text(encoding="utf-8"))
                if description != {"format": "ypuddin-flux3-components", "version": 1, "components": components}:
                    return None
            return dict(components)
        old = context.db.get_kv(_proof_key(root), {})
        marker = root / ".ypuddin-flux3-download.json"
        if not old or not marker.is_file() or marker.stat().st_size > 1024 * 1024:
            return None
        spec = BundleManifest.model_validate(json.loads(marker.read_text(encoding="utf-8"))["manifest"])
        if not available_bundle(context, root, spec):
            return None
        if spec.id == "flux3-base":
            return {"text_encoder": str(root / "text_encoder"), "vae": str(root / "ae.safetensors"),
                    "tokenizer": str(root / "text_encoder")}
        return {"text_encoder": str(root), "tokenizer": str(root)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_components(context):
    """Find existing certified assets, including a BF16 file needing only processor assets."""
    result = {}
    rows = context.db.fetchall("SELECT * FROM models ORDER BY is_default DESC, created_at DESC")
    processors = []
    for stored in context.db.fetchall("SELECT value FROM kv WHERE key LIKE 'model.flux3_components.%'"):
        proof = json.loads(stored["value"])
        if (proof.get("component") == "tokenizer" and proof.get("official_identity") == "flux3-processor-v1"
                and valid_snapshot(context, proof)):
            processors.append(proof["path"])
        if proof.get("component") in {"text_encoder", "vae"} and valid_snapshot(context, proof):
            path = Path(proof["path"])
            rows.append({"kind": proof["component"], "path": str(path / "ae.safetensors")
                         if proof["component"] == "vae" else str(path)})
    for row in rows:
        try:
            path = _safe(context, row["path"])
            if path.is_dir():
                found = resolve_components(context, path)
                if found:
                    for key, value in found.items():
                        result.setdefault(key, value)
                    continue
            if row["kind"] == "vae" and component_sha256(context, path) == VAE_SHA256:
                result.setdefault("vae", str(path))
            elif row["kind"] == "text_encoder":
                certified = context.db.get_kv(proof_key(path), {})
                if not (path.is_file() and component_sha256(context, path) == KREA_TEXT_SHA256
                        or path.is_dir() and certified.get("component") == "text_encoder"
                        and certified.get("official_identity") == "qwen3vl4b-v1"
                        and valid_snapshot(context, certified)):
                    continue
                processor = _component_metadata(context, path).get("processor_path") or (path if path.is_dir() else path.parent)
                processor = _processor(context, processor)
                if processor is None:
                    processor = next((value for candidate in processors if (value := _processor(context, candidate))), None)
                if "text_encoder" not in result or processor and "tokenizer" not in result:
                    result["text_encoder"] = str(path)
                    if processor:
                        result["tokenizer"] = processor
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def available_installation(context, root, spec):
    proof = context.db.get_kv(proof_key(root), {})
    return bool(proof.get("layout_version") == 2 and proof.get("source_manifest_digest") == spec.digest
                and resolve_components(context, root))


def installation_plan(context, storage, spec, *, components=None, authenticate=None):
    from .model_layout import shared_component_folder

    if components is None:
        components = find_components(context)
        authenticate = []
        if "text_encoder" not in components:
            candidate = metadata_candidates(context)
            components.update(candidate)
            if candidate:
                authenticate.append(candidate["text_encoder"])
    else:
        components = dict(components)
    if spec.id == "flux3-text":
        components.pop("vae", None)
    groups = []
    text_files = [item for item in spec.files if item.path.startswith("text_encoder/")]
    if "text_encoder" not in components:
        groups.append(("text_encoder", text_files))
    elif "tokenizer" not in components:
        groups.append(("tokenizer", [item for item in text_files
                                     if not item.path.endswith((".safetensors", ".safetensors.index.json"))]))
    if spec.id == "flux3-base" and "vae" not in components:
        groups.append(("vae", [item for item in spec.files if item.path == "ae.safetensors"]))
    planned = []
    for role, files in groups:
        if not files:
            raise ValueError(f"FLUX.3 source manifest is missing {role} assets")
        identity = hashlib.sha256(json.dumps([item.model_dump() for item in files], sort_keys=True).encode()).hexdigest()
        target = shared_component_folder(storage, role) / f"flux3-{role}-{identity[:12]}"
        planned.append({"role": role, "target": str(target), "files": [item.model_dump() for item in files]})
    own = [item.model_dump() for item in spec.files
           if spec.id == "flux3-base" and not item.path.startswith("text_encoder/") and item.path != "ae.safetensors"]
    bindings = {path: {name: signature(member) for name, member in _files(context, path).items()}
                for path in set(components.values())}
    return {"version": 1, "components": components, "groups": planned, "files": own, "reference_stats": bindings,
            "authenticate": authenticate or []}


def validate_plan(context, storage, spec, plan):
    if not isinstance(plan, dict) or plan.get("version") != 1 or not isinstance(plan.get("components"), dict):
        raise ValueError("invalid FLUX.3 component plan")
    if set(plan["components"]) - {"text_encoder", "vae", "tokenizer"}:
        raise ValueError("invalid FLUX.3 component role")
    if set(plan.get("authenticate", [])) - {plan["components"].get("text_encoder")}:
        raise ValueError("invalid FLUX.3 component authentication plan")
    expected = installation_plan(context, storage, spec, components=plan["components"], authenticate=plan.get("authenticate", []))
    if expected != plan:
        raise ValueError("FLUX.3 component plan or referenced files changed; start a new installation")
    return plan


def _record(context, path, digests):
    files = _files(context, path)
    if set(files) != set(digests):
        raise ValueError("FLUX.3 publication members changed")
    return {"path": str(path), "files": {name: {"sha256": digests[name], "stat": signature(member),
                                               "size": member.stat().st_size} for name, member in files.items()}}


def _json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _validate_group(path, role):
    from ypuddin.models.flux3.latent import Flux3Latent
    from ypuddin.models.flux3.text import Flux3Text

    from .model_inspection import check_flux3_processor

    if role == "text_encoder":
        Flux3Text.validate_paths(path, path)
    elif role == "vae":
        Flux3Latent.validate_path(path / "ae.safetensors")
    else:
        check_flux3_processor(path)


def _group_components(group):
    path, role = Path(group["target"]), group["role"]
    if role == "text_encoder":
        return {"text_encoder": str(path), "tokenizer": str(path)}
    return {role: str(path / "ae.safetensors") if role == "vae" else str(path)}


def _recover_groups(service, row):
    """Keep completed shared assets after interruption; never delete shared paths."""
    from .flux3_downloads import _inside

    context = service.context
    storage = Path(row["storage_root"])
    for group in row["component_plan"]["groups"]:
        target = _inside(storage, Path(group["target"]))
        if not target.exists() or valid_snapshot(context, context.db.get_kv(proof_key(target), {})):
            continue
        marker = target / SHARED
        if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 1024 * 1024:
            continue
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if (saved.get("download_id") != row["id"] or saved.get("component") != group["role"]
                or saved.get("source_manifest") != row["manifest"]):
            continue
        expected = {item["path"].removeprefix("text_encoder/"): item for item in group["files"]}
        if set(_files(context, target)) != set(expected) | {SHARED}:
            continue
        record = certify_component(context, target)
        if any(record["files"][name]["size"] != item["size"] or
               record["files"][name]["sha256"] != saved["files"][name] or
               item.get("sha256") and record["files"][name]["sha256"] != item["sha256"]
               for name, item in expected.items()):
            continue
        _validate_group(target, group["role"])
        context.db.set_kv(proof_key(target), {**record, "component": group["role"], "owner_download_id": row["id"],
                                             "official_identity": "qwen3vl4b-v1" if group["role"] == "text_encoder"
                                             else "flux3-processor-v1" if group["role"] == "tokenizer" else "flux3-vae-v1"})


def registered_under(context, target):
    return any(Path(asset["path"]).resolve().is_relative_to(target)
               for asset in context.db.fetchall("SELECT path FROM models"))


def recover_installation(service, row):
    import shutil

    from .db import now
    from .flux3_downloads import BundleManifest, _inside

    _recover_groups(service, row)
    context, target = service.context, Path(row["target_path"])
    storage = Path(row["storage_root"])
    spec = BundleManifest.model_validate(row["manifest"])
    _inside(storage, target)
    if available_installation(context, target, spec):
        components = resolve_components(context, target)
        primary = str(target) if row["kind"] == "dit" else components["text_encoder"]
        asset = context.db.fetchone("SELECT id FROM models WHERE family='flux3' AND kind=? AND path=?", (row["kind"], primary))
        row.update(status="completed", error=None, model_id=asset["id"] if asset else None,
                   components=components, downloaded_bytes=row["total_bytes"], bytes_per_second=0,
                   eta_seconds=0, finished_at=now())
        return
    marker = target / INSTALLATION
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 1024 * 1024:
        return
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("download_id") != row["id"] or saved.get("source_manifest_digest") != spec.digest:
        return
    files = _files(context, target)
    if set(files) != set(saved["owned_files"]) | {INSTALLATION}:
        return
    if any(signature(files[name]) != item["stat"] for name, item in saved["owned_files"].items()):
        return
    if registered_under(context, target):
        return
    shutil.rmtree(target)


def run_installation(service, id_, storage):
    import shutil
    import time
    import urllib.error
    import urllib.request

    from .db import now
    from .download_errors import download_auth_error, download_http_error
    from .flux3_downloads import BundleFile, BundleManifest, _inside, bundle_source_url, cleanup_stage
    from .model_downloads import _Cancelled, _check_download_header, _Redirect
    from .network import ProxyPolicy
    from .routes_core import ModelBody, add_model, register_flux3_components

    row, event = service.tasks[id_], service.cancelled[id_]
    context, plan = service.context, row["component_plan"]
    spec = BundleManifest.model_validate(row["manifest"])
    stage, target = storage / ".downloads" / id_, Path(row["target_path"])
    policy, token, published, registered = ProxyPolicy(), None, False, False
    done, last, previous = 0, time.monotonic(), 0

    def cancelled():
        if event.is_set() or service.closed:
            raise _Cancelled()

    def transfer(item, destination):
        nonlocal token, policy, done, last, previous
        cancelled()
        policy = ProxyPolicy.from_context(context)
        if token is None:
            token = service.credentials.token(row["provider"], policy=policy) or ""
        headers = {"User-Agent": "YPuddinTrainStudio", "Accept-Encoding": "identity"}
        if token:
            headers["Cookie" if row["provider"] == "modelscope" else "Authorization"] = (
                f"m_session_id={token}" if row["provider"] == "modelscope" else f"Bearer {token}")
        request = urllib.request.Request(bundle_source_url(spec, row["provider"], item.path), headers=headers)
        opener = service.opener or policy.opener(_Redirect())
        destination.parent.mkdir(parents=True, exist_ok=True)
        count, hasher = 0, hashlib.sha256()
        git = hashlib.sha1(f"blob {item.size}\0".encode()) if item.git_blob_sha1 else None
        probe = bytearray() if item.path.endswith(".safetensors") else None
        with opener.open(request, timeout=15) as response, destination.open("xb") as output:
            if response.status != 200 or (response.headers.get("Content-Length")
                                         and int(response.headers["Content-Length"]) != item.size):
                raise ValueError(f"FLUX.3 file size differs: {item.path}")
            while block := response.read(256 * 1024):
                cancelled()
                count += len(block)
                if count > item.size:
                    raise ValueError(f"FLUX.3 file exceeds its pinned size: {item.path}")
                if probe is not None:
                    from .model_inspection import HEADER_LIMIT

                    probe.extend(block[:HEADER_LIMIT + 8 - len(probe)])
                    if len(probe) >= 8:
                        length = int.from_bytes(probe[:8], "little")
                        if not 2 <= length <= HEADER_LIMIT:
                            raise ValueError("invalid safetensors header")
                        if len(probe) >= length + 8:
                            _check_download_header(probe[8:length + 8], "flux3")
                            probe = None
                hasher.update(block)
                if git is not None:
                    git.update(block)
                output.write(block)
                done += len(block)
                current = time.monotonic()
                if current - last >= .4:
                    rate = (done - previous) / (current - last)
                    service._update(id_, downloaded_bytes=done, bytes_per_second=rate,
                                    eta_seconds=max(0, row["total_bytes"] - done) / rate, progress_at=now())
                    last, previous = current, done
            if count != item.size or item.sha256 and hasher.hexdigest() != item.sha256 or git and git.hexdigest() != item.git_blob_sha1:
                raise ValueError(f"FLUX.3 file integrity check failed: {item.path}")
            output.flush()
            os.fsync(output.fileno())
        return hasher.hexdigest()

    try:
        cancelled()
        _inside(storage, stage).mkdir(parents=True, exist_ok=False)
        service._update(id_, status="downloading")
        components = dict(plan["components"])
        snapshots = {}
        for path in components.values():
            if path not in snapshots:
                if path in plan.get("authenticate", []):
                    snapshots[path] = certify_text_component(context, path, event)
                    context.db.set_kv(proof_key(path), snapshots[path])
                else:
                    snapshots[path] = certify_component(context, path, event)
            if {name: item["stat"] for name, item in snapshots[path]["files"].items()} != plan["reference_stats"][path]:
                raise ValueError("FLUX.3 referenced component changed after the download was queued")
        for index, group in enumerate(plan["groups"]):
            cancelled()
            shared = _inside(storage, Path(group["target"]))
            proof = context.db.get_kv(proof_key(shared), {})
            if shared.exists():
                if proof.get("component") != group["role"] or not valid_snapshot(context, proof):
                    raise ValueError("FLUX.3 shared destination exists without an unchanged verified component")
                skipped = sum(item["size"] for item in group["files"])
                row["total_bytes"] -= skipped
                service._update(id_, total_bytes=row["total_bytes"], expected_size=row["total_bytes"])
            else:
                temporary = stage / f"component-{index}"
                temporary.mkdir()
                digests = {}
                for raw in group["files"]:
                    item = BundleFile.model_validate(raw)
                    name = item.path.removeprefix("text_encoder/")
                    digests[name] = transfer(item, _inside(storage, temporary / name))
                _validate_group(temporary, group["role"])
                digests[SHARED] = _json(temporary / SHARED, {"download_id": id_, "component": group["role"],
                                                          "source_manifest": row["manifest"], "files": dict(digests)})
                with service.lock:
                    cancelled()
                    _inside(storage, shared).parent.mkdir(parents=True, exist_ok=True)
                    if shared.exists():
                        raise ValueError("FLUX.3 shared destination appeared during download")
                    temporary.rename(shared)
                proof = {**_record(context, shared, digests), "component": group["role"], "owner_download_id": id_,
                         "official_identity": "qwen3vl4b-v1" if group["role"] == "text_encoder"
                         else "flux3-processor-v1" if group["role"] == "tokenizer" else "flux3-vae-v1"}
                context.db.set_kv(proof_key(shared), proof)
            components.update(_group_components(group))
            snapshots[str(shared)] = {"path": proof["path"], "files": proof["files"]}
        from ypuddin.models.flux3.text import Flux3Text

        Flux3Text.validate_paths(components["text_encoder"], components["tokenizer"])
        own = stage / "installation"
        own.mkdir()
        digests = {}
        for raw in plan["files"]:
            item = BundleFile.model_validate(raw)
            digests[item.path] = transfer(item, _inside(storage, own / item.path))
        if row["kind"] == "dit":
            from ypuddin.models.flux3.latent import Flux3Latent
            from ypuddin.models.flux3.loading import transformer_config

            transformer_config(own / "model.safetensors")
            Flux3Latent.validate_path(components["vae"])
            digests[DESCRIPTION] = _json(own / DESCRIPTION, {"format": "ypuddin-flux3-components", "version": 1,
                                                          "components": components})
        references = {}
        for role, path in components.items():
            if path not in snapshots:
                snapshots[path] = certify_component(context, path, event)
            references[role] = snapshots[path]
            if not valid_snapshot(context, references[role]):
                raise ValueError("FLUX.3 referenced component changed during installation")
        receipt = {"layout_version": 2, "download_id": id_, "source_manifest_digest": spec.digest,
                   "components": components, "references": references, "owned_files": _record(context, own, digests)["files"]}
        digests[INSTALLATION] = _json(own / INSTALLATION, receipt)
        with service.lock:
            cancelled()
            _inside(storage, target).parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("FLUX.3 installation destination already exists")
            own.rename(target)
            published = True
        proof = {**receipt, "owned": _record(context, target, digests)}
        with context.db.transaction():
            context.db.set_kv(proof_key(target), proof)
            register_flux3_components(target, context, row["is_default"])
            if row["kind"] == "dit":
                asset = add_model(ModelBody(family="flux3", kind="dit", path=str(target), dtype=row["dtype"],
                                            is_default=row["is_default"]), context)
            else:
                asset = context.db.fetchone("SELECT id FROM models WHERE family='flux3' AND kind='text_encoder' AND path=?",
                                             (components["text_encoder"],))
                if asset is None:
                    raise ValueError("FLUX.3 component registration was not completed")
        registered = True
        service._update(id_, status="completed", model_id=asset["id"], components=components, downloaded_bytes=done,
                        bytes_per_second=0, eta_seconds=0, finished_at=now())
    except _Cancelled:
        service._update(id_, status="cancelled", finished_at=now())
    except Exception as error:
        if published and not registered:
            # Only this small installation directory is owned; references remain untouched.
            with context.db.transaction():
                if not registered_under(context, target):
                    shutil.rmtree(_inside(storage, target))
        message = policy.redact(error)
        if token:
            message = message.replace(token, "[redacted]")
        if isinstance(error, urllib.error.HTTPError):
            message = download_auth_error(error, provider=row["provider"], authenticated=bool(token), mirror=False) or download_http_error(error.code)
        service._update(id_, status="cancelled" if event.is_set() else "failed", error=None if event.is_set() else message, finished_at=now())
    finally:
        try:
            cleanup_stage(storage, id_)
        except (OSError, ValueError):
            pass
