# EmoSens

`emosens.py` adapts muooon's EmoSens v3.9.3 from the `v3.9.0_ecc` branch,
fixed at commit `bc925f9ab366a2fc8b20e5b5205d7082a8097c98`:

- [Algorithm](https://github.com/muooon/EmoSens/blob/bc925f9ab366a2fc8b20e5b5205d7082a8097c98/optimizer/emosens.py), Git blob `14ea7c25d1f80bb40b6f65f7937e3329e5075cac`.
- [License](https://github.com/muooon/EmoSens/blob/bc925f9ab366a2fc8b20e5b5205d7082a8097c98/LICENSE), Apache-2.0, reproduced in `EMOSENS_LICENSE.txt`.
- [Author's usage recommendations](https://github.com/muooon/EmoSens/blob/bc925f9ab366a2fc8b20e5b5205d7082a8097c98/README.md).

The loss EMA, scalar controller, shared emoPulse, uncorrected first/second moments,
decoupled weight decay and optional shadow blend retain the pinned algorithm.
Studio changes the integration as follows:

- Removes the import-time global `Tensor.backward` replacement. The Trainer supplies
  the unscaled, sample-weighted loss for each update, after gradient validation.
  Gradient accumulation uses the whole update's loss; DDP uses the existing global
  sample-count/loss reduction. A closure may also provide the loss explicitly.
- Keeps moments, optional shadow and low-precision master weights in FP32. FP32
  parameters use the author's update directly; FP16/BF16 parameters receive a copy
  of the updated FP32 master. No additional optimizer wrapper or CPU offload is used.
- Preserves zero-rate frozen groups and skips controller advancement when no active
  parameter has a gradient. Nonzero per-group learning-rate overrides are rejected.
- Stores the controller and loss EMA separately from per-parameter state, includes
  initial settings and a format version, validates parameter/settings identity,
  and restores FP32 tensors without changing the caller's checkpoint dictionary.
- Reports each completed update's emoPulse as its learning rate. External schedules
  and warmup are disabled. The configured learning rate is an initial scale.
- Logs a stability notification only when its condition becomes true. `stopcoef`
  and `notify` do not stop training. `use_shadow` remains off by default.

New configurations follow the pinned author's architecture/mode recommendations:
Transformer/DiT adapter scale `0.1`, full scale `0.01`; SDXL U-Net adapter scale `1.0`,
full scale `0.1`. Explicit saved values are retained. The constructor's upstream
default is `1.0`; these scales are not actual first-step learning rates.

The implementation uses only the existing PyTorch dependency and Python standard
library. FSDP remains outside the existing supported optimizer list. The author's
README correction labels older theory/experiment material as unverified; no speed,
convergence or image-quality claim is adopted here.
