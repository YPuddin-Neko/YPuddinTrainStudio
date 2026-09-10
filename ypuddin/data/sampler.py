"""Deterministic, resumable bucket batch sampler (never trains an image twice per epoch)."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, bucket_of: list[tuple[int, int]], batch_size: int, *, seed: int = 0, drop_last: bool = False, world_size: int = 1, rank: int = 0):
        self.bucket_of = list(bucket_of)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.position = 0  # number of batches already consumed in the current epoch
        self._plan_cache: tuple[int, list[list[int]]] | None = None

    # ----------------------------------------------------------------- planning
    def plan(self, epoch: int | None = None) -> list[list[int]]:
        epoch = self.epoch if epoch is None else epoch
        if self._plan_cache and self._plan_cache[0] == epoch:
            return self._plan_cache[1]
        rng = random.Random(self.seed * 1_000_003 + epoch)
        by_bucket: dict[tuple[int, int], list[int]] = {}
        for i, b in enumerate(self.bucket_of):
            by_bucket.setdefault(b, []).append(i)
        batches: list[list[int]] = []
        for key in sorted(by_bucket):
            idx = by_bucket[key]
            rng.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                chunk = idx[s : s + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        rng.shuffle(batches)
        if self.world_size > 1:
            # equal share per rank; drop the tail so every rank sees the same number of steps
            usable = len(batches) - len(batches) % self.world_size
            batches = batches[self.rank : usable : self.world_size]
        self._plan_cache = (epoch, batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch = epoch
            self.position = 0

    def set_position(self, position: int) -> None:
        self.position = int(position)

    def __iter__(self) -> Iterator[list[int]]:
        plan = self.plan()
        for pos in range(self.position, len(plan)):
            yield plan[pos]

    def __len__(self) -> int:
        return len(self.plan())

    def batches_per_epoch(self) -> int:
        return len(self.plan(0))

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = int(state.get("seed", self.seed))
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self._plan_cache = None
