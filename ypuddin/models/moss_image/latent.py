"""MOSS-Image and full FLUX.2 autoencoders, FP32 mean encoding and frozen BN."""
from pathlib import Path

import torch
from safetensors import safe_open

from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.flux2.latent import (  # noqa: F401
    Flux2Latent,
    convert_vae_state_dict,
    patchify,
    unpatchify,
)
from ypuddin.runtime_imports import LIBRARY_IMPORTS

from .loading import read_json, validate_weights, weight_files

VAE_CHANNELS = (128, 256, 512, 512)
DECODER_CHANNELS = (96, 192, 384, 384)


class MOSSImageLatent(Flux2Latent):
    @staticmethod
    def vae_config(path):
        path = Path(path)
        config_file = (path if path.is_dir() else path.parent) / "config.json"
        return read_json(config_file) if path.is_dir() or config_file.exists() else {}

    @classmethod
    def validate_path(cls, path):
        with LIBRARY_IMPORTS:
            from diffusers import AutoencoderKLFlux2

        path = Path(path)
        config = cls.vae_config(path)
        decoder = tuple(config.get("decoder_block_out_channels") or config.get("block_out_channels", VAE_CHANNELS))
        if ((config and config.get("_class_name") != "AutoencoderKLFlux2")
                or config.get("latent_channels", 32) != 32
                or tuple(config.get("block_out_channels", VAE_CHANNELS)) != VAE_CHANNELS
                or decoder not in {DECODER_CHANNELS, VAE_CHANNELS}
                or tuple(config.get("patch_size", (2, 2))) != (2, 2) or config.get("quantization_config")):
            raise ValueError("MOSS-Image 需要配套的小解码器 VAE 或标准 FLUX.2 VAE。")
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            model = AutoencoderKLFlux2.from_config(config)
        expected = model.state_dict()
        if config.get("decoder_block_out_channels") and decoder == DECODER_CHANNELS:
            validate_weights(path, expected, basename="diffusion_pytorch_model", integer_keys={"bn.num_batches_tracked"})
        else:
            headers = {}
            files, mapping = weight_files(path, "diffusion_pytorch_model")
            for file in files:
                with safe_open(str(file), framework="pt", device="cpu") as weights:
                    for key in weights.keys():
                        tensor = weights.get_slice(key)
                        allowed = {"BF16", "F16", "F32"} | ({"I64", "I32"} if key == "bn.num_batches_tracked" else set())
                        if (key in headers or tensor.get_dtype() not in allowed
                                or mapping is not None and (key not in mapping or (path / mapping[key]).resolve() != file.resolve())):
                            raise ValueError(f"MOSS-Image VAE 权重类型或分片索引无效：{key}")
                        headers[key] = torch.empty(tuple(tensor.get_shape()), device="meta")
            if mapping is not None and set(mapping) != set(headers):
                raise ValueError("MOSS-Image VAE 权重分片缺少索引中的张量。")
            if path.is_dir() and "encoder.down.0.block.0.norm1.weight" in headers:
                raise ValueError("BFL 格式 VAE 请选择 safetensors 文件；VAE 目录需要 Diffusers 格式。")
            converted = convert_vae_state_dict(headers, model.config)
            if (len(converted) != len(headers) or converted.keys() != expected.keys()
                    or any(converted[key].shape != expected[key].shape for key in expected)
                    or sum(t.numel() for t in converted.values()) != sum(t.numel() for t in headers.values())):
                raise ValueError("MOSS-Image VAE 权重与标准 FLUX.2 配置不符。")
        return path

    def __init__(self, path, *, device="cpu"):
        super().__init__(self.validate_path(path), device=device)
        config_file = (self.path if self.path.is_dir() else self.path.parent) / "config.json"
        small = tuple(self.config.get("decoder_block_out_channels") or ()) == DECODER_CHANNELS
        self.fingerprint = content_fingerprint(
            [self.path, *([config_file] if config_file.is_file() else [])],
            namespace=f"moss-image-{'small' if small else 'full'}-vae-mode-fp32-patch2-bn-v1",
        )
