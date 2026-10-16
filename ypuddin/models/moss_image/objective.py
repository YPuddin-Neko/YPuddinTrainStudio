"""Released stage-three discrete timestep law (MOSS-Image report, Appendix B.1)."""
import math

import torch

from ypuddin.objectives.flow import PackedFlowObjective, TimestepSampler

from .latent import patchify, unpatchify


def stage_three_pmf():
    edges = torch.arange(1001, dtype=torch.float64) * (0.95 / 1000)
    z = (torch.log((1 - edges) / edges) + 1.09) / 0.95
    cdf = 0.5 * torch.erfc(z / math.sqrt(2))
    return 0.99 * cdf.diff() / cdf[-1] + 0.01 / 1000


class MOSSImageTimestepSampler(TimestepSampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        grid = torch.arange(1, 1001, dtype=torch.float64) / 1000
        included = (grid >= cfg.t_min) & (grid <= cfg.t_max)
        if not included.any():
            raise ValueError("MOSS-Image 时间步范围内至少需要一个 0.001 间隔的离散点。")
        self.grid = grid[included].float()
        probability = stage_three_pmf()[included]
        self.cdf = (probability / probability.sum()).cumsum(0)
        self.cdf[-1] = 1.0

    def _base_from_uniform(self, u):
        return self.grid.to(u.device)[torch.searchsorted(self.cdf.to(u.device), u.double()).clamp(max=len(self.grid) - 1)]

    def _apply_shift(self, t, num_tokens):
        # Bounds condition the discrete PMF; sigma=1 remains a valid endpoint.
        return t


class MOSSImageObjective(PackedFlowObjective):
    def __init__(self, cfg):
        super().__init__(cfg, pack=patchify, unpack=unpatchify)
        if cfg.timestep_sampling == "moss_image":
            self.sampler = MOSSImageTimestepSampler(cfg)

    def prepare(self, x0, t, *, generator=None):
        # Noise is drawn in FP32; spatial options act before packing 2x2 positions.
        return super().prepare(x0.float(), t, generator=generator)
