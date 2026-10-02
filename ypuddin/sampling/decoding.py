"""Preview decoding with the selected sampler's latent normalization order."""


def decode_preview(pipeline, latents, *, noise: str = "comfyui"):
    if noise == "comfyui" and hasattr(pipeline, "decode_comfy"):
        return pipeline.decode_comfy(latents)
    return pipeline.decode(latents)
