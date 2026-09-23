# Klein attention compatibility

`attention.py` adapts the two Flux2 attention processors from Hugging Face
Diffusers 0.40.0, Copyright 2025 Black Forest Labs, The HuggingFace Team and
The InstantX Team. All rights reserved. Licensed under Apache License 2.0.
The upstream license is preserved in `LICENSE-DIFFUSERS.txt`.

Source: https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/transformer_flux2.py

The adaptation preserves projection, normalization, rotary embedding, joint
text/image ordering and output projection. It replaces attention dispatch with
the installed HIP vendor's public FlashAttention function and adds explicit
validation and model-local installation/restoration. It does not bundle the
FlashAttention implementation or pretrained model weights.

Model configuration and tokenizer asset notices are in `assets/NOTICE.md`.
