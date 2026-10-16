"""Exact shared MOSS-Image identities and immutable component references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .flux3_components import _files, _record, _safe, signature
from .flux3_components import valid_snapshot as _valid_snapshot

DESCRIPTION = ".ypuddin-moss-image-components.json"
MARKER = ".ypuddin-moss-image-download.json"


def valid_snapshot(context, record):
    return _valid_snapshot(context, record) and all(_valid_snapshot(context, item) for item in record.get("sidecars", []))


def proof_key(path):
    return "model.moss_image." + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()


def component_files(role):
    from .moss_image_downloads import bundle_manifest

    spec = bundle_manifest("moss-image-text" if role == "text_encoder" else "moss-image-vae")
    return [item for item in spec.files if item.path != "LICENSE"]


def component_identity(role):
    values = [(item.path, item.size, item.sha256) for item in component_files(role)]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def validate_component(role, components):
    if role == "text_encoder":
        from ypuddin.models.moss_image.text import MOSSImageText

        MOSSImageText.validate_paths(components["text_encoder"], components["tokenizer"])
    else:
        from ypuddin.models.moss_image.latent import MOSSImageLatent

        MOSSImageLatent.validate_path(components["vae"])


def component_paths(root, role):
    return ({"text_encoder": str(root / "text_encoder"), "tokenizer": str(root / "tokenizer")}
            if role == "text_encoder" else {"vae": str(root / "vae")})


def _expected(components, role):
    expected = {}
    for item in component_files(role):
        key, name = item.path.split("/", 1)
        base = Path(components[key])
        if key == "vae" and base.is_file():
            target = base if name.endswith(".safetensors") else base.parent / name
        else:
            target = base / name
        expected[target] = item
    return expected


def candidate(context, components, role):
    """Only metadata is inspected here; explicit preparation authenticates bytes."""
    try:
        expected = _expected(components, role)
        actual = {member for key in ({"text_encoder", "tokenizer"} if role == "text_encoder" else {"vae"})
                  for member in _files(context, components[key]).values()}
        if role == "vae" and Path(components["vae"]).is_file():
            actual.add(_safe(context, Path(components["vae"]).parent / "config.json"))
        return actual == set(expected) and all(_safe(context, path).stat().st_size == item.size
                                              for path, item in expected.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def authenticate(context, components, role, event=None, *, owner=None):
    """Authenticate registered candidates only during an explicit import or preparation."""
    from .model_downloads import _Cancelled

    if not candidate(context, components, role):
        raise ValueError("MOSS-Image 组件文件与固定版本不符。")
    before = {str(path): signature(path) for path in _expected(components, role)}
    for path, item in _expected(components, role).items():
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                if event is not None and event.is_set():
                    raise _Cancelled()
                hasher.update(block)
        if hasher.hexdigest() != item.sha256 or signature(path) != before[str(path)]:
            raise ValueError("MOSS-Image 组件校验失败，原文件已保留。")
    validate_component(role, components)
    return certify_transferred(context, components, role, owner=owner)


def certify_transferred(context, components, role, *, owner):
    records = {}
    for key in ({"text_encoder", "tokenizer"} if role == "text_encoder" else {"vae"}):
        path = Path(components[key])
        digests = {item.path.split("/", 1)[1]: item.sha256 for item in component_files(role)
                   if item.path.startswith(key + "/")}
        if key == "vae" and path.is_file():
            records[key] = _record(context, path, {"": digests["diffusion_pytorch_model.safetensors"]})
            records[key]["sidecars"] = [_record(context, path.parent / "config.json", {"": digests["config.json"]})]
        else:
            records[key] = _record(context, path, digests)
    value = {"identity": component_identity(role), "role": role, "records": records,
             "owner_download_id": owner}
    for record in records.values():
        context.db.set_kv(proof_key(record["path"]), value)
    return value


def valid_component(context, proof, role):
    return (proof.get("identity") == component_identity(role) and proof.get("role") == role
            and set(proof.get("records", {})) == ({"text_encoder", "tokenizer"} if role == "text_encoder" else {"vae"})
            and all(valid_snapshot(context, item) for item in proof["records"].values()))


def component_source(context, path):
    proof = context.db.get_kv(proof_key(path), {})
    if proof.get("role") in {"text_encoder", "vae"} and valid_component(context, proof, proof["role"]):
        return proof.get("owner_download_id")
    if proof.get("owned") and valid_snapshot(context, proof["owned"]):
        return proof.get("download_id")
    return None


def find_components(context, *, candidates=False):
    from .model_recommendations import component_metadata

    result = {}
    rows = context.db.fetchall("SELECT * FROM models WHERE family='moss_image' ORDER BY is_default DESC, created_at DESC")
    paths = [(row["kind"], Path(row["path"])) for row in rows if row["kind"] in {"text_encoder", "vae"}]
    storage = Path(context.settings()["paths"]["models_dir"]).resolve()
    from .model_layout import shared_component_folder

    for role in ("text_encoder", "vae"):
        paths.append((role, shared_component_folder(storage, role) / ("moss-" + component_identity(role)[:16]) / role))
    for role, path in paths:
        if role in result:
            continue
        proof = context.db.get_kv(proof_key(path), {})
        if valid_component(context, proof, role):
            result.update({key: value["path"] for key, value in proof["records"].items()})
        elif candidates:
            components = {role: str(path)}
            if role == "text_encoder":
                components["tokenizer"] = component_metadata(context, path).get("processor_path") or str(path.parent / "tokenizer")
            if candidate(context, components, role):
                result.update(components)
    return result


def reference_records(context, components):
    result = {}
    for role in ("text_encoder", "vae"):
        if role not in components:
            continue
        proof = context.db.get_kv(proof_key(components[role]), {})
        if not valid_component(context, proof, role):
            raise ValueError("MOSS-Image 共享组件已变化，请重新校验。")
        result.update(proof["records"])
    return result


def resolve_components(context, root):
    try:
        root = _safe(context, root)
        proof = context.db.get_kv(proof_key(root), {})
        if not valid_snapshot(context, proof["owned"]):
            return None
        components, references = proof["components"], proof["references"]
        if set(components) != {"text_encoder", "vae", "tokenizer"} or set(references) != set(components):
            return None
        if any(references[key]["path"] != path or not valid_snapshot(context, references[key])
               for key, path in components.items()):
            return None
        if proof.get("layout_version") == 1:
            return dict(components)
        if json.loads((root / DESCRIPTION).read_text()) != {
            "format": "ypuddin-moss-image-components", "version": 1, "components": components
        }:
            return None
        return dict(components)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def manifest_identity(spec):
    return hashlib.sha256(json.dumps([(item.path, item.size, item.sha256) for item in spec.files],
                                     separators=(",", ":")).encode()).hexdigest()


def authenticate_root(context, root):
    """Recognize an imported official HF root without manufacturing a download origin."""
    from .moss_image_downloads import bundle_manifest

    root = _safe(context, root)
    if (root / DESCRIPTION).exists():
        return
    spec = bundle_manifest("moss-image-base")
    expected = {item.path: item for item in spec.files}
    files = _files(context, root)
    if not set(expected) <= set(files) or any(files[name].stat().st_size != item.size for name, item in expected.items()):
        return
    previous = context.db.get_kv(proof_key(root), {})
    if previous.get("official_identity") == manifest_identity(spec) and resolve_components(context, root):
        return
    components = {"text_encoder": str(root / "text_encoder"), "tokenizer": str(root / "tokenizer"), "vae": str(root / "vae")}
    references = reference_records(context, components)
    before = {name: signature(path) for name, path in files.items()}
    for name, item in expected.items():
        prefix, _, relative = name.partition("/")
        if prefix in references and references[prefix]["files"][relative]["sha256"] == item.sha256:
            continue
        digest = hashlib.sha256()
        with files[name].open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != item.sha256:
            raise ValueError("MOSS-Image 主模型与固定官方版本不符，原文件已保留。")
    if set(_files(context, root)) != set(files) or any(signature(path) != before[name] for name, path in files.items()):
        raise ValueError("MOSS-Image 模型在校验期间发生变化。")
    owned = {"path": str(root), "files": {name: {"stat": before[name], "size": before[name][0],
             "sha256": expected[name].sha256 if name in expected else None} for name in files}}
    context.db.set_kv(proof_key(root), {"layout_version": 1, "official_identity": manifest_identity(spec),
                                      "components": components, "references": references, "owned": owned})
