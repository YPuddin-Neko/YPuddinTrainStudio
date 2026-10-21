# Muon

The Newton–Schulz kernel and EMA momentum update are adapted from
[KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py),
revision `f98f1cacc0263b04290753e32be8d498c1efc806`.

Matrix groups use BF16 Newton–Schulz iterations with coefficients
`(3.4445, -4.7750, 2.0315)`, Frobenius normalization by `norm + 1e-7`,
and update scaling `sqrt(max(1, rows / columns))`. Momentum is
`m = beta*m + (1-beta)*gradient`; Nesterov uses
`(1-beta)*gradient + beta*m`. Fused projections are processed as complete
matrices. Input/output projections and nonmatrix parameters can be assigned
to the auxiliary AdamW groups by the parameter grouper.

Integration choices:

- Momentum and AdamW moments stay in FP32. FP16/BF16 parameters also retain
  FP32 master weights; FP32 parameters need no master copy.
- A missing gradient or zero group learning rate leaves weights and state
  unchanged. Zero-valued gradients still apply momentum and weight decay.
- Gradients are not modified by the optimizer. All active gradients and the
  next Muon BF16 inputs are checked for finite values before the step changes
  any parameter or state.
- Each group's learning rate is already scaled by the parameter grouper.
  Weight decay uses that rate, before the Muon matrix shape adjustment.
- Checkpoints bind parameter names, shapes, dtypes, roles, base learning rates,
  and algorithm settings. Scheduled learning rates are restored separately.
- This implementation uses ordinary PyTorch operations available in the
  project's minimum PyTorch version; it does not depend on `torch.optim.Muon`.

The upstream MIT license follows.

MIT License

Copyright (c) 2024 Keller Jordan

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
