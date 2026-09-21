"""Real CPU/Gloo protocol validation; intentionally not Windows or CUDA acceptance."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from ypuddin.train.gloo_probe import _exercise_gloo, _select_probe_dtypes


def main():
    rank = int(os.environ["RANK"])
    rng = torch.get_rng_state().clone()
    store, store_rank, world = next(dist.rendezvous("env://", timeout=timedelta(seconds=15)))
    assert store_rank == rank and world == 2
    dist.init_process_group(
        "gloo", store=store, rank=store_rank, world_size=world, timeout=timedelta(seconds=30)
    )
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=15))
        try:
            # Simulated heterogeneous capability values are exchanged over a
            # real CPU process group. No CUDA support is inferred from this.
            selected = _select_probe_dtypes(group, (torch.float32,), supports_bf16=rank == 0)
            assert selected == (torch.float32, torch.float16, torch.float64)
            try:
                _select_probe_dtypes(group, (torch.bfloat16,), supports_bf16=rank == 0)
            except ValueError as exc:
                assert "rank [1]" in str(exc)
            else:
                raise AssertionError("all ranks must reject unsupported required BF16")
            result = _exercise_gloo(torch.device("cpu"), group, (torch.float32,))
            result["mixed_capability_control_passed"] = True
        finally:
            dist.destroy_process_group(group)
        assert torch.equal(rng, torch.get_rng_state())
        result.update(
            cpu_only=True,
            cuda_validated=False,
            windows_validated=False,
            rng_unchanged=True,
            public_rendezvous_used=True,
        )
    finally:
        dist.destroy_process_group()
    result["default_group_destroyed"] = not dist.is_initialized()
    (Path(sys.argv[1]) / f"probe-rank-{rank}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
