# FLUX 3 Image Klein source attribution

`autoencoder.py` is an unchanged copy of Black Forest Labs' source at
[`45e57c9fb06f9751cdced29a65c5d2171003be05`](https://github.com/black-forest-labs/flux-3-image-klein/tree/45e57c9fb06f9751cdced29a65c5d2171003be05/src/flux3).
`dit.py` is adapted from that revision. Its changes expose configurable dimensions,
add non-reentrant block checkpointing, allow PyTorch SDPA backend selection, and
calculate RoPE frequencies on CPU when the target is MPS. Default weight names,
shapes, normalization and velocity output follow the upstream implementation.

Those source files are distributed under Apache License 2.0; the upstream license
is included as `LICENSE-BFL.txt`. No model weights are included in this package.
The official Base weights at
[`black-forest-labs/flux-3-image-klein-base`](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)
are separately governed by FLUX Kommunity License v1.0, with their own access and
license terms. Source licensing does not change the model weight license.
