# Standard DoRA fused computation

The comparison source is LyCORIS 4.1.3, commit
`7abd5709f74f9159246ff9c6cc83bc0d5118b639`, licensed under Apache-2.0:

- https://github.com/KohakuBlueleaf/LyCORIS/tree/7abd5709f74f9159246ff9c6cc83bc0d5118b639
- `lycoris/kernels/triton/dora/decompose.py`
- `lycoris/kernels/autograd/dora.py`
- License: https://github.com/KohakuBlueleaf/LyCORIS/blob/7abd5709f74f9159246ff9c6cc83bc0d5118b639/LICENSE.md

The local standard-mode kernels implement the FP32 equation directly and are
not an unmodified copy of the LyCORIS kernel. They retain the raw norm `r` from
forward and differentiate `m * W / (r + epsilon)` using the factor
`r * (r + epsilon)^2`, with the norm's zero-vector subgradient. They do not use
the reference kernel's `(r + epsilon)^3` denominator or its mixed-dtype promotion.

This path applies to standard linear DoRA with strength 1. ComfyUI decomposition
remains a separate implementation with its own axis, precision and strength rules.
