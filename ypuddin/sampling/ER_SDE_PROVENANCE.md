# ER-SDE sampling provenance

The implementation in `er_sde.py` derives its Taylor update from Qinpeng Cui's
MIT-licensed [ER-SDE-Solver](https://github.com/QinpengCui/ER-SDE-Solver), pinned
to commit `25ca6c2e6b065754a55694dc6941a3259fa0f082`, specifically
[`vp_3_order_taylor`](https://github.com/QinpengCui/ER-SDE-Solver/blob/25ca6c2e6b065754a55694dc6941a3259fa0f082/er_sde_solver.py)
and the author's type-7 noise function. The original MIT notice is retained in
`ER_SDE_LICENSE.txt`. No AnimaLoraStudio or ComfyUI solver code was copied.

The primary paper is Cui, Zhang, Bao and Liao,
[Elucidating the Solution Space of Extended Reverse-Time SDE for Diffusion Models](https://openaccess.thecvf.com/content/WACV2025/html/Cui_Elucidating_the_Solution_Space_of_Extended_Reverse-Time_SDE_for_Diffusion_WACV_2025_paper.html),
WACV 2025, pp. 243–252. Its benchmark results are not measurements of this trainer.

This adaptation uses rectified-flow time `t`, `alpha = 1-t`,
`lambda = t/(1-t)` and velocity-derived denoised predictions `x-t*v`. A strictly
decreasing schedule starts below one; the terminal zero step returns the current
denoised estimate directly. The multistep history starts at first order, then
second and third order as enough predictions become available.

The author's `phi(lambda) = lambda * (exp(lambda**0.3) + 10)` is evaluated through
log ratios. Its two integrals are expressed as bounded moments on a unit interval
and computed with adaptive float64 Gauss–Legendre quadrature. This replaces the
reference's fixed left-rectangle approximation and avoids constructing overflowing
exponentials or subtracting almost equal noise variances. Tensor accumulation uses
FP32, or FP64 when explicitly requested. Every Gaussian draw uses a private CPU
generator; sampling does not advance training's global RNG. Nonfinite predictions
and invalid schedules raise errors instead of silently replacing values.

Focused tests compare complete trajectories against separately implemented author
equations with dense trapezoidal integrals, analytic constant/linear denoisers,
nonuniform and extreme schedules, CFG, cancellation and RNG isolation. These are
numerical and integration checks; they do not establish large-model image quality,
CUDA performance or bitwise equivalence with other applications.
