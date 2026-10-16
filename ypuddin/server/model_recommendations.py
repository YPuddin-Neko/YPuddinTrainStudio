"""Download sources and pinned metadata for the model catalog.

Source metadata checked 2026-09-13 using the Hugging Face tree API and ModelScope
repo/files API. Provider mappings are explicit; a matching repository name is never
assumed by the client. Updated upstream bytes must pass the pinned SHA-256 check.
Illustrious v0.1 and Krea Raw BF16 ModelScope sources rechecked 2026-09-15.
Klein Base 4B/9B, Qwen3 4B/8B and FLUX.2 VAE ModelScope sources checked 2026-09-23.
NoobAI-XL V-Pred 1.0 and 1.1 sources checked 2026-09-27. ModelScope has 1.1 in the official LaxharLAB
repository; V-Pred 1.0 only in the ModelE mirror, whose file is byte-identical (same SHA-256).
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .errors import ApiError
from .flux3_downloads import BundleId, BundleManifest, bundle_manifest
from .model_credentials import Provider
from .moss_image_downloads import MossBundleId, MossBundleManifest


class RecommendedSource(BaseModel):
    provider: Provider
    repo_id: str
    filename: str
    revision: str
    url: str
    manifest: BundleManifest | MossBundleManifest | None = None


class RecommendedModel(BaseModel):
    id: str
    family: Literal["anima", "krea2", "sdxl", "flux", "flux2", "flux3", "moss_image"]
    kind: Literal["dit", "text_encoder", "text_encoder_2", "vae"]
    name: str
    dtype: Literal["bf16", "fp16", "fp32", "fp8"]
    size: int = Field(gt=0)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    bundle: BundleId | MossBundleId | None = None
    manifest: BundleManifest | MossBundleManifest | None = None
    recommended: bool = True
    purpose: Literal["training", "inference"] = "training"
    variant: Literal["raw", "turbo"] | None = None
    sources: list[RecommendedSource]
    model_id: str | None = None
    available_path: str | None = None
    processor_path: str | None = None
    active_download_id: str | None = None
    is_default: bool = False

    @model_validator(mode="after")
    def integrity(self):
        if self.bundle is None:
            if self.sha256 is None or self.manifest is not None:
                raise ValueError("single-file recommendations require a pinned SHA-256")
        elif (self.family not in {"flux3", "moss_image"} or self.manifest is None or self.manifest.id != self.bundle
              or self.size != self.manifest.size or self.sha256 is not None):
            raise ValueError("directory recommendations require their matching file manifest")
        for source in self.sources:
            if source.manifest is not None and (
                self.bundle is None or source.manifest.id != self.bundle or source.manifest.size != self.size
                or source.manifest.repo_id != source.repo_id or source.manifest.revision != source.revision
            ):
                raise ValueError("source manifest must match its directory recommendation and source")
        return self


def _entry(
    id_,
    family,
    kind,
    name,
    dtype,
    size,
    sha256,
    repo,
    filename,
    recommended=True,
    *,
    purpose="training",
    variant=None,
    modelscope_source: RecommendedSource | None = None,
):
    return RecommendedModel(
        id=id_,
        family=family,
        kind=kind,
        name=name,
        dtype=dtype,
        size=size,
        sha256=sha256,
        recommended=recommended,
        purpose=purpose,
        variant=variant,
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id=repo,
                filename=filename,
                revision="main",
                url=f"https://huggingface.co/{repo}/blob/main/{filename}",
            ),
            modelscope_source
            or RecommendedSource(
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


def _flux3_entry(id_):
    manifest = bundle_manifest(id_)
    sources = []
    for provider in ("huggingface", "modelscope"):
        spec = bundle_manifest(id_, provider)
        sources.append(RecommendedSource(
            provider=provider, repo_id=spec.repo_id, filename="" if id_ == "flux3-base" else "text_encoder",
            revision=spec.revision, manifest=spec,
            url=f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}" if provider == "huggingface"
                else f"https://modelscope.cn/models/{spec.repo_id}/files",
        ))
    return RecommendedModel(
        id=id_, family="flux3", kind="dit" if id_ == "flux3-base" else "text_encoder",
        name="FLUX.3 Image Klein Base" if id_ == "flux3-base" else "FLUX.3 Qwen3-VL-4B",
        dtype="bf16", size=manifest.size, bundle=id_, manifest=manifest, sources=sources,
    )


def _moss_entry(id_):
    from .moss_image_downloads import bundle_manifest

    bundle = "moss-image-dit" if id_ == "moss-image-base" else id_
    manifest = bundle_manifest(bundle)
    kind, name, dtype = {
        "moss-image-base": ("dit", "MOSS-Image Base 5B", "bf16"),
        "moss-image-text": ("text_encoder", "MOSS-Image Qwen3.5-9B", "bf16"),
        "moss-image-vae": ("vae", "MOSS-Image FLUX.2 Small Decoder", "fp32"),
    }[id_]
    sources = []
    for provider in ("huggingface", "modelscope"):
        spec = bundle_manifest(bundle, provider)
        sources.append(RecommendedSource(provider=provider, repo_id=spec.repo_id,
                       filename="transformer" if kind == "dit" else kind, revision=spec.revision, manifest=spec,
                       url=f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}" if provider == "huggingface"
                       else f"https://modelscope.cn/models/{spec.repo_id}/files"))
    return RecommendedModel(id=id_, family="moss_image", kind=kind, name=name, dtype=dtype,
                            size=manifest.size, bundle=bundle, manifest=manifest, sources=sources)


RECOMMENDATIONS = [
    *[_moss_entry(id_) for id_ in ("moss-image-base", "moss-image-text", "moss-image-vae")],
    *[_flux3_entry(id_) for id_ in ("flux3-base", "flux3-text")],
    RecommendedModel(
        id="flux2-klein-base-4b",
        family="flux2",
        kind="dit",
        name="FLUX.2 Klein Base 4B · BF16",
        dtype="bf16",
        size=7751105712,
        sha256="9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="black-forest-labs/FLUX.2-klein-base-4B",
                filename="flux-2-klein-base-4b.safetensors",
                revision="a3b4f4849157f664bdbc776fd7453c2783562f4d",
                url="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/blob/a3b4f4849157f664bdbc776fd7453c2783562f4d/flux-2-klein-base-4b.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="black-forest-labs/FLUX.2-klein-base-4B",
                filename="flux-2-klein-base-4b.safetensors",
                revision="384cd13880a8dd205782035743eead9ec374df5c",
                url="https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B/files",
            ),
        ],
    ),
    RecommendedModel(
        id="flux2-klein-base-9b",
        family="flux2",
        kind="dit",
        name="FLUX.2 Klein Base 9B · BF16",
        dtype="bf16",
        size=18157185168,
        sha256="4a54fad7f5f741b99eee217198daac20b8d8e515e2a1f5b064fd51cf074f95bd",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="black-forest-labs/FLUX.2-klein-base-9B",
                filename="flux-2-klein-base-9b.safetensors",
                revision="32773329fbe7e81a90ef971740e8ba4b0364ecf3",
                url="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/32773329fbe7e81a90ef971740e8ba4b0364ecf3/flux-2-klein-base-9b.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="black-forest-labs/FLUX.2-klein-base-9B",
                filename="flux-2-klein-base-9b.safetensors",
                revision="332e29c86837a0143b3984ebda6f875fd9df76ab",
                url="https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-9B/files",
            ),
        ],
    ),
    RecommendedModel(
        id="flux2-qwen3-4b",
        family="flux2",
        kind="text_encoder",
        name="Qwen3 4B",
        dtype="bf16",
        size=8044982048,
        sha256="6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="Comfy-Org/flux2-klein-4B",
                filename="split_files/text_encoders/qwen_3_4b.safetensors",
                revision="5f526678002e43af5551dadb73ce2e8c91b43afe",
                url="https://huggingface.co/Comfy-Org/flux2-klein-4B/blob/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/text_encoders/qwen_3_4b.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="Comfy-Org/flux2-klein-4B",
                filename="split_files/text_encoders/qwen_3_4b.safetensors",
                revision="305618a79d0d7b2b167266c8a4082ee99e948ca2",
                url="https://modelscope.cn/models/Comfy-Org/flux2-klein-4B/files",
            ),
        ],
    ),
    RecommendedModel(
        id="flux2-qwen3-8b",
        family="flux2",
        kind="text_encoder",
        name="Qwen3 8B",
        dtype="bf16",
        size=16381517176,
        sha256="f0ff9239d56269ca1d05e5f86da6a79fac111af464955681f11c7ab0ec5ef6c1",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="Comfy-Org/flux2-klein-9B",
                filename="split_files/text_encoders/qwen_3_8b.safetensors",
                revision="3f62d9d8ae1fec33c6e91453d5c712855b096b55",
                url="https://huggingface.co/Comfy-Org/flux2-klein-9B/blob/3f62d9d8ae1fec33c6e91453d5c712855b096b55/split_files/text_encoders/qwen_3_8b.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="Comfy-Org/flux2-klein-9B",
                filename="split_files/text_encoders/qwen_3_8b.safetensors",
                revision="0bb21a1a5059384ac93399fc1f31e752e605d1ee",
                url="https://modelscope.cn/models/Comfy-Org/flux2-klein-9B/files",
            ),
        ],
    ),
    RecommendedModel(
        id="flux2-vae",
        family="flux2",
        kind="vae",
        name="FLUX.2 VAE · Klein 4B / 9B 共用",
        dtype="fp32",
        size=336211292,
        sha256="868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="Comfy-Org/flux2-klein-4B",
                filename="split_files/vae/flux2-vae.safetensors",
                revision="5f526678002e43af5551dadb73ce2e8c91b43afe",
                url="https://huggingface.co/Comfy-Org/flux2-klein-4B/blob/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/vae/flux2-vae.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="Comfy-Org/flux2-klein-4B",
                filename="split_files/vae/flux2-vae.safetensors",
                revision="305618a79d0d7b2b167266c8a4082ee99e948ca2",
                url="https://modelscope.cn/models/Comfy-Org/flux2-klein-4B/files",
            ),
        ],
    ),
    *[
        RecommendedModel(
            id=f"krea2-turbo-{dtype}",
            family="krea2",
            kind="dit",
            name=f"Krea 2 Turbo · {label} · 仅采样",
            dtype=dtype,
            size=size,
            sha256=digest,
            purpose="inference",
            variant="turbo",
            recommended=dtype == "fp8",
            sources=[
                RecommendedSource(
                    provider="huggingface",
                    repo_id=KREA,
                    filename=f"diffusion_models/{filename}",
                    revision="e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96",
                    url=f"https://huggingface.co/{KREA}/blob/e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96/diffusion_models/{filename}",
                )
            ],
        )
        for dtype, label, filename, size, digest in (
            (
                "bf16",
                "BF16",
                "krea2_turbo_bf16.safetensors",
                26283332608,
                "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3",
            ),
            (
                "fp8",
                "FP8 scaled",
                "krea2_turbo_fp8_scaled.safetensors",
                13141730784,
                "eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1",
            ),
        )
    ],
    RecommendedModel(
        id="sdxl-illustrious-v01",
        family="sdxl",
        kind="dit",
        name="Illustrious-XL（光辉）v0.1",
        dtype="fp16",
        size=6938040760,
        sha256="3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
                filename="Illustrious-XL-v0.1.safetensors",
                revision="f08f0826ffe32183ba2d1f4106dd5b32e195a02e",
                url="https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/f08f0826ffe32183ba2d1f4106dd5b32e195a02e/Illustrious-XL-v0.1.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="OnomaAIResearch/Illustrious-xl-early-release-v0",
                filename="Illustrious-XL-v0.1.safetensors",
                revision="6f7fa36d9cb8aede0e05eeb8790966151a0d21a0",
                url="https://modelscope.cn/models/OnomaAIResearch/Illustrious-xl-early-release-v0/files",
            ),
        ],
    ),
    RecommendedModel(
        id="sdxl-noobai-vpred-10",
        family="sdxl",
        kind="dit",
        name="NoobAI-XL V 预测 1.0",
        dtype="bf16",
        recommended=False,
        size=7105350110,
        sha256="ea349eeae87ca8d25ba902c93810f7ca83e5c82f920edf12f273af004ae02819",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="Laxhar/noobai-XL-Vpred-1.0",
                filename="NoobAI-XL-Vpred-v1.0.safetensors",
                revision="66aa55e3469c27c29a89813cd35dd95fb7485fa1",
                url="https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/66aa55e3469c27c29a89813cd35dd95fb7485fa1/NoobAI-XL-Vpred-v1.0.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="ModelE/noobai-XL-Vpred-1.0",
                filename="NoobAI-XL-Vpred-v1.0.safetensors",
                revision="647714aee089d7977e2e835142852ff8d3c2c6e0",
                url="https://modelscope.cn/models/ModelE/noobai-XL-Vpred-1.0/files",
            ),
        ],
    ),
    RecommendedModel(
        id="sdxl-noobai-eps-11",
        family="sdxl",
        kind="dit",
        name="NoobAI-XL E 预测 1.1",
        dtype="bf16",
        recommended=False,
        size=7105349958,
        sha256="6681e8e4b134c81f16533acedb0d406d7e5e366e1624b4105178c64d00b05d51",
        sources=[
            RecommendedSource(
                provider="huggingface",
                repo_id="Laxhar/noobai-XL-1.1",
                filename="NoobAI-XL-v1.1.safetensors",
                revision="814a274af2b8097c0828819d561ec74c7d0c6cea",
                url="https://huggingface.co/Laxhar/noobai-XL-1.1/blob/814a274af2b8097c0828819d561ec74c7d0c6cea/NoobAI-XL-v1.1.safetensors",
            ),
            RecommendedSource(
                provider="modelscope",
                repo_id="LaxharLAB/NoobAI-XL",
                filename="file.safetensors",
                revision="97675079e70111933c737a2dc1928ef2d4e51977",
                url="https://modelscope.cn/models/LaxharLAB/NoobAI-XL/files",
            ),
        ],
    ),
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
        variant="raw",
        modelscope_source=RecommendedSource(
            provider="modelscope",
            repo_id="krea/Krea-2-Raw",
            filename="raw.safetensors",
            revision="4c6e0f5dee814bea2d3f5e1fe7264e78a42c4370",
            url="https://modelscope.cn/models/krea/Krea-2-Raw/files",
        ),
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
        variant="raw",
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


# The FLUX.3 AE is byte-identical to the already-pinned FLUX.2 VAE.
RECOMMENDATIONS.insert(2, next(entry for entry in RECOMMENDATIONS if entry.id == "flux2-vae").model_copy(
    update={"id": "flux3-vae", "family": "flux3", "name": "FLUX.3 VAE"}, deep=True,
))


def find_recommendation(id_: str) -> RecommendedModel | None:
    return next((item for item in RECOMMENDATIONS if item.id == id_), None)


def model_display_name(context, row) -> str | None:
    """Name verified MOSS component directories while retaining custom basenames."""
    if row.get("family") != "moss_image":
        return None
    recommendation = {"dit": "moss-image-base", "text_encoder": "moss-image-text", "vae": "moss-image-vae"}
    entry = find_recommendation(recommendation.get(row.get("kind"), ""))
    if entry is None or entry.manifest is None:
        return None
    from .moss_image_components import proof_key
    from .moss_image_downloads import available_bundle, bundle_manifest

    try:
        path = Path(row["path"]).expanduser().absolute()
        if path.resolve() != path or not path.is_dir():
            return None
        kind = row["kind"]
        if kind == "dit":
            if path.name != "transformer" and not any(
                path.name == bundle_manifest(bundle, provider).digest[:12]
                for bundle in ("moss-image-base", "moss-image-dit")
                for provider in ("huggingface", "modelscope")
            ):
                return None
        else:
            proof = context.db.get_kv(proof_key(path), {})
            if path.name != kind or proof.get("records", {}).get(kind, {}).get("path") != str(path):
                return None
        if available_bundle(context, path, entry.manifest):
            return entry.name
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


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
    from ypuddin.models.krea2.variants import record_verified_variant

    record_verified_variant(path, sha256)


def _bundle_manifests(entry):
    for source in entry.sources:
        if source.manifest is not None:
            yield source.manifest
        elif entry.manifest is not None and (source.repo_id, source.revision) == (entry.manifest.repo_id, entry.manifest.revision):
            yield entry.manifest


def verify_local_model(context, entry: RecommendedModel, path: Path) -> None:
    """Verify a local candidate once before choosing it as this exact recommendation."""
    if entry.id in {"flux3-text", "krea2-qwen3vl"}:
        from .flux3_components import find_components

        shared = find_components(context)
        if shared.get("text_encoder") == str(path.resolve()):
            if entry.family == "flux3":
                from ypuddin.models.flux3.text import Flux3Text

                if not shared.get("tokenizer"):
                    raise ApiError("Qwen3-VL processor 资源不完整。", code="model.component", status=422)
                try:
                    Flux3Text.validate_paths(path, Path(shared["tokenizer"]))
                except (ValueError, OSError) as error:
                    raise ApiError(str(error), code="model.component", status=422) from error
            return
    if entry.bundle:
        from .model_downloads import _bundle_module

        available_bundle = _bundle_module(entry.family).available_bundle
        if not any(available_bundle(context, path, spec) for spec in _bundle_manifests(entry)):
            raise ApiError("模型目录包的文件已变化或缺少完整下载记录。", code="model.integrity")
        return
    if not context.is_allowed(path):
        raise ApiError("model path is outside allowed storage roots", code="model.path", status=403)
    try:
        signature = _signature(path)
        if signature[0] != entry.size:
            raise ApiError("local model size does not match the recommendation", code="model.integrity")
        from .flux3_components import component_sha256

        digest = component_sha256(context, path)
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
        remember_verified_file(context, path, digest)
    except OSError:
        raise ApiError("cannot read the local model for verification", code="model.read") from None


def component_key(path: Path) -> str:
    return "model.component." + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def _component_signature(path: Path, processor: Path | None = None) -> str:
    from .model_status import signature

    data = [signature(path), signature(processor) if processor else None]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _weight_signature(path: Path) -> dict | None:
    if path.is_file():
        return {"": _signature(path)}
    if path.is_dir():
        files = sorted(member for member in path.rglob("*") if member.is_file()
                       and member.name.endswith((".safetensors", ".safetensors.index.json")))
        return {member.relative_to(path).as_posix(): _signature(member) for member in files} or None
    return None


def component_source_download(context, path: Path) -> str | None:
    """Resolve the task that transferred these weights without reading their payloads."""
    from .flux3_components import proof_key, valid_snapshot
    from .flux3_downloads import BundleManifest, available_bundle

    try:
        path = path.resolve()
        if not context.is_allowed(path) or not path.exists():
            return None
        from .moss_image_components import component_source

        if source := component_source(context, path):
            return source
        current = _weight_signature(path)
        if current is None:
            return None
        stored = context.db.get_kv(component_key(path), {})
        if stored.get("source_signature") == current and stored.get("source_download_id"):
            return stored["source_download_id"]
        shared = path if path.is_dir() else path.parent
        proof = context.db.get_kv(proof_key(shared), {})
        role = proof.get("component")
        if ((role == "text_encoder" and path == shared or role == "vae" and path == shared / "ae.safetensors")
                and proof.get("owner_download_id") and valid_snapshot(context, proof)):
            return proof["owner_download_id"]
        if (path.is_dir() and proof.get("layout_version") == 2
                and "model.safetensors" in proof.get("owned", {}).get("files", {})
                and valid_snapshot(context, proof["owned"])):
            return proof.get("download_id")
        for task in context.db.get_kv("model_downloads", []):
            if task.get("status") != "completed":
                continue
            target = Path(task["target_path"])
            if not task.get("bundle"):
                if (path == target and path.is_file() and task.get("sha256")
                        and cached_sha256(context, path) == task["sha256"]):
                    return task["id"]
            elif task.get("layout_version", 1) == 1:
                members = {target} if task["kind"] == "text_encoder" else {
                    target, target / "text_encoder", target / "ae.safetensors"}
                if path in members and available_bundle(context, target, BundleManifest.model_validate(task["manifest"])):
                    return task["id"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def remember_component(context, path: Path, *, processor_path=None, source_download_id=None) -> None:
    processor = Path(processor_path).resolve() if processor_path else None
    previous = component_metadata(context, path)
    value = {"processor_path": str(processor) if processor else previous.get("processor_path"),
             "source_download_id": source_download_id or component_source_download(context, path),
             "source_signature": _weight_signature(path)}
    processor = Path(value["processor_path"]) if value["processor_path"] else None
    value["signature"] = _component_signature(path, processor)
    context.db.set_kv(component_key(path), value)


def component_metadata(context, path: Path) -> dict:
    value = context.db.get_kv(component_key(path), {})
    if not value:
        return {}
    processor = Path(value["processor_path"]) if value.get("processor_path") else None
    if (not path.exists() or not context.is_allowed(path.resolve())
        or processor and (not processor.is_dir() or not context.is_allowed(processor.resolve()))):
        return {}
    if value.get("signature") != _component_signature(path, processor):
        return {}
    return value


def related_download(entry, tasks, *, bundle=None):
    """A shared component reports the task that owns its actual preparation."""
    shared_text = {"krea2-qwen3vl", "flux3-text"}
    shared_vae = {"flux2-vae", "flux3-vae"}
    for row in tasks:
        if row.get("status") not in {"queued", "downloading"}:
            continue
        owner = row.get("recommendation_id") or row.get("bundle")
        if owner == entry.id:
            return row
        moss = {"moss-image-base", "moss-image-text", "moss-image-vae"}
        if entry.id in moss and owner in moss and (
                (bundle or entry.bundle) == "moss-image-base" or row.get("bundle") == "moss-image-base"):
            return row
        if entry.sha256 and row.get("sha256") == entry.sha256 and row.get("kind") == entry.kind:
            return row
        if entry.id in shared_text and (owner in shared_text or row.get("bundle") == "flux3-base"):
            return row
        if entry.id in shared_vae and (owner in shared_vae or row.get("bundle") == "flux3-base"):
            return row
        if entry.id == "flux3-base" and owner in shared_text | shared_vae:
            return row
    return None


def available_models(context, *, tasks=None) -> list[RecommendedModel]:
    from .flux3_components import component_sha256, find_components, resolve_components
    from .flux3_downloads import available_bundle

    assets = context.db.fetchall("SELECT * FROM models ORDER BY is_default DESC, created_at DESC")
    candidates = list(assets)
    for asset in assets:
        if asset["family"] != "flux3" or asset["kind"] != "dit":
            continue
        components = resolve_components(context, Path(asset["path"])) or {}
        for kind in ("vae", "text_encoder"):
            if components.get(kind) and not any(row["path"] == components[kind] and row["kind"] == kind for row in candidates):
                candidates.append({"id": None, "family": "flux3", "kind": kind, "path": components[kind],
                                   "is_default": False, "processor_path": components.get("tokenizer")})
    shared = find_components(context)
    tasks = tasks if tasks is not None else context.db.get_kv("model_downloads", [])
    from .moss_image_components import find_components as find_moss_components
    from .moss_image_downloads import available_bundle as available_moss_bundle

    moss_shared = find_moss_components(context)
    result = []
    for entry in RECOMMENDATIONS:
        own = None
        available = None
        verified_paths = set()
        processor = None
        if entry.id in {"flux3-text", "krea2-qwen3vl"} and shared.get("text_encoder"):
            path = shared["text_encoder"]
            processor = shared.get("tokenizer")
            if entry.family != "flux3" or processor:
                own = next((asset for asset in assets if asset["family"] == entry.family
                            and asset["kind"] == entry.kind and asset["path"] == path), None)
                available = own or {"path": path}
                verified_paths.add(path)
        if entry.family == "moss_image" and entry.kind in moss_shared:
            path = moss_shared[entry.kind]
            processor = moss_shared.get("tokenizer") if entry.kind == "text_encoder" else None
            own = next((asset for asset in assets if asset["family"] == entry.family
                        and asset["kind"] == entry.kind and asset["path"] == path), None)
            available = own or {"path": path}
            verified_paths.add(path)
        if available is None and entry.bundle:
            specs = tuple(_bundle_manifests(entry))
            own = next((asset for asset in assets if asset["family"] == entry.family and asset["kind"] == entry.kind
                        and any((available_moss_bundle if entry.family == "moss_image" else available_bundle)(context, Path(asset["path"]), spec) for spec in specs)), None)
            available = own
            if own:
                verified_paths.add(own["path"])
                if entry.kind == "text_encoder":
                    processor = component_metadata(context, Path(own["path"])).get("processor_path") or own["path"]
        elif available is None:
            filenames = {Path(source.filename).name for source in entry.sources}
            matches = []
            for asset in candidates:
                if asset["kind"] != entry.kind or (entry.kind != "vae" and asset["family"] != entry.family):
                    continue
                path = Path(asset["path"])
                try:
                    if not context.is_allowed(path) or not path.is_file() or path.stat().st_size != entry.size:
                        continue
                    digest = component_sha256(context, path)
                    if digest == entry.sha256 or digest is None and path.name in filenames:
                        matches.append(asset)
                        if digest == entry.sha256:
                            verified_paths.add(asset["path"])
                except OSError:
                    continue
            own = next((asset for asset in matches if asset.get("id") and asset["family"] == entry.family), None)
            available = own or next(iter(matches), None)
        task = related_download(entry, tasks)
        result.append(entry.model_copy(update={
            "model_id": own["id"] if own else None,
            "available_path": available["path"] if available else None,
            "processor_path": processor,
            "active_download_id": task["id"] if task else None,
            "is_default": bool(entry.purpose == "training" and own and own["is_default"]
                               and own["path"] in verified_paths),
        }))
    return result
