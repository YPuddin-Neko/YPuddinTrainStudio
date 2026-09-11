"""Training-compatible single-file weights, independently verified against both publishers.

Source metadata checked 2026-09-12 using the Hugging Face tree API and ModelScope
repo/files API. Provider mappings are explicit; a matching repository name is never
assumed by the client. Updated upstream bytes must pass the pinned SHA-256 check.
"""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .errors import ApiError
from .model_credentials import Provider


class RecommendedSource(BaseModel):
    provider: Provider
    repo_id: str
    filename: str
    revision: str
    url: str


class RecommendedModel(BaseModel):
    id: str
    family: Literal["anima", "krea2"]
    kind: Literal["dit", "text_encoder", "vae"]
    name: str
    dtype: Literal["bf16", "fp8"]
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recommended: bool = True
    sources: list[RecommendedSource]
    model_id: str | None = None
    available_path: str | None = None
    is_default: bool = False


def _entry(id_, family, kind, name, dtype, size, sha256, repo, filename, recommended=True):
    return RecommendedModel(
        id=id_,
        family=family,
        kind=kind,
        name=name,
        dtype=dtype,
        size=size,
        sha256=sha256,
        recommended=recommended,
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id=repo,
                filename=filename,
                revision="main",
                url=f"https://huggingface.co/{repo}/blob/main/{filename}",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id=repo,
                filename=filename,
                revision="master",
                url=f"https://modelscope.cn/models/{repo}/files",
            ),
        ],
    )


ANIMA = "circlestone-labs/Anima"
KREA = "Comfy-Org/Krea-2"
RECOMMENDATIONS = [
    _entry(
        "anima-base-1",
        "anima",
        "dit",
        "Anima Base 1.0",
        "bf16",
        4182218328,
        "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",
        ANIMA,
        "split_files/diffusion_models/anima-base-v1.0.safetensors",
    ),
    _entry(
        "anima-preview3",
        "anima",
        "dit",
        "Anima Preview 3 Base",
        "bf16",
        4182218360,
        "14fffe8ad5116cd73b9a4696f6a89d7e5f6efdd24b2e4785603aa891a9b2295b",
        ANIMA,
        "split_files/diffusion_models/anima-preview3-base.safetensors",
        False,
    ),
    _entry(
        "anima-qwen3",
        "anima",
        "text_encoder",
        "Qwen3 0.6B Base",
        "bf16",
        1192135096,
        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        ANIMA,
        "split_files/text_encoders/qwen_3_06b_base.safetensors",
    ),
    _entry(
        "krea2-raw-bf16",
        "krea2",
        "dit",
        "Krea 2 Raw · BF16",
        "bf16",
        26283332608,
        "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7",
        KREA,
        "diffusion_models/krea2_raw_bf16.safetensors",
    ),
    _entry(
        "krea2-raw-fp8",
        "krea2",
        "dit",
        "Krea 2 Raw · FP8 scaled",
        "fp8",
        13141730784,
        "48cd5d6c100297968349b41a8e77c6591d1dac18a215807f5f25f59e5c54cd61",
        KREA,
        "diffusion_models/krea2_raw_fp8_scaled.safetensors",
        False,
    ),
    _entry(
        "krea2-qwen3vl",
        "krea2",
        "text_encoder",
        "Qwen3-VL 4B · BF16",
        "bf16",
        8875719384,
        "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34",
        KREA,
        "text_encoders/qwen3vl_4b_bf16.safetensors",
    ),
    *[
        _entry(
            f"{family}-vae",
            family,
            "vae",
            "Qwen Image VAE",
            "bf16",
            253806246,
            "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
            ANIMA,
            "split_files/vae/qwen_image_vae.safetensors",
        )
        for family in ("anima", "krea2")
    ],
]


def find_recommendation(id_: str) -> RecommendedModel | None:
    return next((item for item in RECOMMENDATIONS if item.id == id_), None)


def _signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


def _verification_key(path: Path) -> str:
    return "model.sha256." + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def cached_sha256(context, path: Path) -> str | None:
    """Read stat-validated proof without hashing multi-GB assets on each catalog GET."""
    row = context.db.get_kv(_verification_key(path), {})
    return row.get("sha256") if row.get("signature") == _signature(path) else None


def remember_verified_file(context, path: Path, sha256: str) -> None:
    """Record a completed download's already-verified bytes after their final rename."""
    context.db.set_kv(_verification_key(path), {"signature": _signature(path), "sha256": sha256})


def verify_local_model(context, entry: RecommendedModel, path: Path) -> None:
    """Verify a local candidate once before choosing it as this exact recommendation."""
    if not context.is_allowed(path):
        raise ApiError("model path is outside allowed storage roots", code="model.path", status=403)
    try:
        signature = _signature(path)
        if signature[0] != entry.size:
            raise ApiError("local model size does not match the recommendation", code="model.integrity")
        digest = cached_sha256(context, path)
        if digest is None:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if _signature(path) != signature:
                raise ApiError(
                    "local model changed during verification; retry after writes finish",
                    code="model.changed",
                    status=409,
                )
            digest = hasher.hexdigest()
            context.db.set_kv(_verification_key(path), {"signature": signature, "sha256": digest})
        if digest != entry.sha256:
            raise ApiError(
                "local model SHA-256 does not match the recommendation; the file was preserved",
                code="model.integrity",
            )
    except OSError:
        raise ApiError("cannot read the local model for verification", code="model.read") from None


def available_models(context) -> list[RecommendedModel]:
    assets = context.db.fetchall("SELECT * FROM models ORDER BY is_default DESC, created_at DESC")
    result = []
    for entry in RECOMMENDATIONS:
        # Unknown local files are only candidates until /use verifies their contents.
        # Completed downloads already have hash proof, so list refreshes remain inexpensive.
        filenames = {Path(source.filename).name for source in entry.sources}
        matches = []
        verified_paths = set()
        for asset in assets:
            if asset["kind"] != entry.kind or (entry.kind != "vae" and asset["family"] != entry.family):
                continue
            path = Path(asset["path"])
            try:
                if (
                    path.name in filenames
                    and context.is_allowed(path)
                    and path.is_file()
                    and path.stat().st_size == entry.size
                ):
                    digest = cached_sha256(context, path)
                    if digest in (None, entry.sha256):
                        matches.append(asset)
                        if digest == entry.sha256:
                            verified_paths.add(asset["path"])
            except OSError:
                continue
        own = next((asset for asset in matches if asset["family"] == entry.family), None)
        available = own or next(iter(matches), None)
        result.append(
            entry.model_copy(
                update={
                    "model_id": own["id"] if own else None,
                    "available_path": available["path"] if available else None,
                    "is_default": bool(
                        own
                        and own["is_default"]
                        and own["path"] in verified_paths
                    ),
                }
            )
        )
    return result
