# Automagic

`automagic.py` adapts Automagic v1 from Ostris' AI Toolkit:

- Source: https://github.com/ostris/ai-toolkit/blob/2b4c525489c6fd42a8e99724d9c528b3dcbecb92/toolkit/optimizers/automagic.py
- License: https://github.com/ostris/ai-toolkit/blob/2b4c525489c6fd42a8e99724d9c528b3dcbecb92/LICENSE
- Source revision: `2b4c525489c6fd42a8e99724d9c528b3dcbecb92`.

This is the **FP32-state implementation** of the original v1 update rule: factored
second moments, RMS clipping, and elementwise learning rates adjusted by update
sign agreement. It is not Automagic v2 or a fused-backward optimizer.

Changes made for this trainer:

- Ordinary `step()` with no backward hooks; gradient accumulation and external
  gradient unscaling happen before the optimizer reads gradients.
- Learning-rate masks and moments use FP32 instead of upstream's quantized mask.
  This uses 3 additional bytes per trainable parameter for the mask and avoids
  its quantization error; trajectories and memory do not exactly match upstream.
- FP16/BF16 trainable parameters use an FP32 master copy (4 additional bytes per
  parameter) so small updates are retained. FP32 parameters need no master copy.
- States are native tensors. Resume preserves their dtypes and maps them by
  saved parameter-group IDs even when some parameters have never had gradients.
- Parameter swapping and trainable Quanto tensors are not supported. Frozen
  quantized base models do not pass their weights to this optimizer.
- Group-specific initialization and bounds are honored, invalid values are
  rejected, closures enable autograd, and reporting returns element-weighted
  mean adaptive learning rates rather than the initial learning rate.

## MIT License

Copyright (c) 2024 Ostris, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
