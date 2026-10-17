"""MOSS-Image's small-decoder Flux2 autoencoder, FP32 mean encoding and frozen BN."""
from pathlib import Path

import torch

from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.flux2.latent import Flux2Latent, patchify, unpatchify  # noqa: F401
from ypuddin.runtime_imports import LIBRARY_IMPORTS

from .loading import read_json, validate_weights

VAE_CHANNELS = (128, 256, 512, 512)
DECODER_CHANNELS = (96, 192, 384, 384)


class MOSSImageLatent(Flux2Latent):
    @classmethod
    def validate_path(cls, path):
        with LIBRARY_IMPORTS:
            from diffusers import AutoencoderKLFlux2

        path = Path(path)
        config = read_json((path if path.is_dir() else path.parent) / "config.json")
        if (config.get("_class_name") != "AutoencoderKLFlux2" or config.get("latent_channels") != 32
                or tuple(config.get("block_out_channels", ())) != VAE_CHANNELS
                or tuple(config.get("decoder_block_out_channels", ())) != DECODER_CHANNELS
                or config.get("patch_size") != [2, 2] or config.get("quantization_config")):
            raise ValueError("MOSS-Image 需要配套的 32 通道、小解码器 AutoencoderKLFlux2。")
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            expected = AutoencoderKLFlux2.from_config(config).state_dict()
        validate_weights(path, expected, basename="diffusion_pytorch_model", integer_keys={"bn.num_batches_tracked"})
        return path

    def __init__(self, path, *, device="cpu"):
        super().__init__(self.validate_path(path), device=device)
        self.fingerprint = content_fingerprint([self.path, (self.path if self.path.is_dir() else self.path.parent) / "config.json"],
                                               namespace="moss-image-small-vae-mode-fp32-patch2-bn-v1")
