"""Third-party model code vendored for the Anima family (Apache-2.0; provenance in ``NOTICE.md``).

Kept import-light on purpose: importing this package pulls in nothing but ``torch``/``einops``/``numpy``/
``safetensors``.  Import the submodules you need::

    from ypuddin.models.anima.vendor.cosmos_dit import Anima, LLMAdapter, ANIMA_2B_CONFIG, infer_dit_config
    from ypuddin.models.anima.vendor.qwen_image_vae import AutoencoderKLQwenImage, load_vae
    from ypuddin.models.anima.vendor.qwen_image_vae_2d import AutoencoderKLQwenImage2D
"""
