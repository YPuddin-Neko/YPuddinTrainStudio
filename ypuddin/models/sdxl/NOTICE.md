# SDXL model integration

The implementation uses installed Hugging Face Diffusers and Transformers APIs.
No AnimaLoraStudio or sd-scripts implementation is copied into this module.

Small architecture and CLIP tokenizer assets in `assets/` come from the official
[Stability AI SDXL base repository](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b).
`assets/manifest.json` records the source revision, byte lengths and SHA-256 digests.
No pretrained neural-network weights are bundled. The upstream model repository
identifies its model license as CreativeML Open RAIL++-M; see its model card/license.
The upstream license text is preserved in `LICENSE-SDXL.txt`.

Reference APIs:

- [Diffusers single-file loading](https://huggingface.co/docs/diffusers/v0.40.0/en/api/loaders/single_file)
- [Official SDXL pipeline conditioning](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py)

All runtime component and tokenizer loading uses local files only. A standard SDXL
base checkpoint supplies its embedded UNet, CLIP-L, CLIP-G and VAE directly. External
component paths take precedence. Diffusers directories need safetensors weights;
standalone `.ckpt`/`.pt`/`.bin` component loading uses `torch.load(weights_only=True)`.
Custom single-file geometries may provide component configs beside the checkpoint
in `sdxl_config/{unet,text_encoder,text_encoder_2,vae}/config.json`. Tokenizer overrides
must contain both `tokenizer/` and `tokenizer_2/`; otherwise the bundled CLIP tokenizers
are used. Refiner-only and inpainting UNets are rejected.
