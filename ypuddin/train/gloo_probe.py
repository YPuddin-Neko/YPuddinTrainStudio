"""Per-job CUDA/Gloo verification before Windows DDP loads any training assets.

Build availability alone does not prove that a Windows wheel implements CUDA
collectives. Run the operations on the selected devices and a small real DDP
graph. FSDP is deliberately outside this contract.
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class _ProbeModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        # Avoid consuming CPU or CUDA RNG before training restores/seeds it.
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=torch.float32))

    def forward(self, inputs):
        return inputs @ self.weight.T


def _exercise_gloo(device: torch.device, group, dtypes: tuple[torch.dtype, ...]) -> dict:
    """The same protocol can be tested with CPU Gloo without faking CUDA tensors."""
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    checked = []
    operation = "initialization"
    try:
        for dtype in dtypes:
            operation = f"broadcast/{dtype}"
            value = torch.tensor([1.0, -2.0] if rank == 0 else [0.0, 0.0], device=device, dtype=dtype)
            dist.broadcast(value, src=0, group=group)
            expected = torch.tensor([1.0, -2.0], device=device, dtype=dtype)
            if not torch.equal(value, expected):
                raise RuntimeError("broadcast result differs")
            checked.append(operation)

            operation = f"all_reduce/{dtype}"
            value = torch.full((2,), rank + 1, device=device, dtype=dtype)
            dist.all_reduce(value, group=group)
            if not torch.equal(value, torch.full_like(value, world * (world + 1) // 2)):
                raise RuntimeError("all_reduce result differs")
            checked.append(operation)

        for name, reduction, expected in (("MAX", dist.ReduceOp.MAX, world), ("MIN", dist.ReduceOp.MIN, 1)):
            operation = f"all_reduce/int64/{name}"
            count = torch.tensor([rank + 1], device=device, dtype=torch.int64)
            dist.all_reduce(count, op=reduction, group=group)
            if count.item() != expected:
                raise RuntimeError("global count/finite flag differs")
            checked.append(operation)

        operation = "all_gather/float32"
        value = torch.tensor([rank], device=device, dtype=torch.float32)
        gathered = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(gathered, value, group=group)
        if [item.item() for item in gathered] != list(range(world)):
            raise RuntimeError("all_gather rank values differ")
        checked.append(operation)

        operation = "all_gather_object/control"
        objects = [None] * world
        dist.all_gather_object(objects, {"rank": rank}, group=group)
        if objects != [{"rank": index} for index in range(world)]:
            raise RuntimeError("control-plane rank values differ")
        checked.append(operation)

        operation = "DDP constructor/forward/backward/update"
        model = _ProbeModel(device)
        ddp = DistributedDataParallel(
            model,
            process_group=group,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.25)
        inputs = torch.full((1, 2), rank + 1.0, device=device, dtype=torch.float32)
        ddp(inputs).sum().backward()
        expected_grad = torch.full_like(model.weight, (world + 1) / 2)
        if model.weight.grad is None or not torch.equal(model.weight.grad, expected_grad):
            raise RuntimeError("DDP did not average gradients correctly")
        before = model.weight.detach().clone()
        optimizer.step()
        if not torch.equal(model.weight, before - 0.25 * expected_grad):
            raise RuntimeError("DDP optimizer update differs")
        replicas = [torch.empty_like(model.weight) for _ in range(world)]
        dist.all_gather(replicas, model.weight.detach(), group=group)
        if not all(torch.equal(model.weight, replica) for replica in replicas):
            raise RuntimeError("DDP replicas differ after the optimizer update")
        checked.append(operation)
        if device.type == "cuda":
            operation = "CUDA synchronize"
            torch.cuda.synchronize(device)
            checked.append(operation)
        return {
            "backend": "gloo",
            "device": str(device),
            "rank": rank,
            "world_size": world,
            "checks": checked,
        }
    except Exception as exc:
        raise RuntimeError(
            f"多卡通信检查失败：rank {rank} / {operation}: {exc}。尚未加载训练模型；请检查 Windows PyTorch/Gloo 与所选显卡。"
        ) from exc


def _select_probe_dtypes(group, required_dtypes: tuple[torch.dtype, ...], *, supports_bf16: bool):
    """Negotiate on CPU before any dtype-dependent CUDA collective sequence."""
    ranks = [None] * dist.get_world_size(group)
    required = sorted({str(dtype) for dtype in required_dtypes})
    dist.all_gather_object(ranks, {"bf16": supports_bf16, "required": required}, group=group)
    if any(rank["required"] != required for rank in ranks):
        raise ValueError("多卡通信检查失败：各 rank 请求的训练精度不同，尚未执行 CUDA 集体通信。")
    dtypes = (torch.float32, torch.float16, torch.float64)
    if torch.bfloat16 in required_dtypes:
        unsupported = [index for index, rank in enumerate(ranks) if not rank["bf16"]]
        if unsupported:
            raise ValueError(
                f"多卡通信检查失败：rank {unsupported} 的显卡不支持本次训练所需的 BF16，请改用 FP16 或 FP32。"
            )
        dtypes += (torch.bfloat16,)
    return dtypes


def probe_windows_cuda_ddp(
    device: torch.device | str, *, required_dtypes: tuple[torch.dtype, ...] = ()
) -> dict:
    """Verify CUDA collectives and DDP on every rank, with a bounded subgroup.

    The caller owns the default process group and must destroy it on failure.
    This function owns only its subgroup and does not write files or load assets.
    Test FP32/FP16 gradients and FP64 global loss totals; BF16 only when required.
    """
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Windows DDP communication check requires a rank-specific CUDA device")
    if not dist.is_initialized() or dist.get_backend() != "gloo" or dist.get_world_size() < 2:
        raise ValueError("Windows DDP communication check requires an initialized multi-rank Gloo group")
    if any(dtype not in (torch.float32, torch.float16, torch.bfloat16) for dtype in required_dtypes):
        raise ValueError("Windows DDP communication check received an unsupported training dtype")
    print(
        json.dumps(
            {
                "type": "distributed.probe",
                "phase": "checking_communication",
                "message": "正在检查多卡通信",
                "device": str(device),
                "backend": "gloo",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    try:
        dtypes = _select_probe_dtypes(group, required_dtypes, supports_bf16=torch.cuda.is_bf16_supported())
        result = _exercise_gloo(device, group, dtypes)
    except BaseException:
        # A failed backend may also fail teardown; preserve the useful operation
        # error while the caller/torchrun cleans the default group and siblings.
        with suppress(Exception):
            dist.destroy_process_group(group)
        raise
    else:
        dist.destroy_process_group(group)
    print(json.dumps({"type": "distributed.probe_passed", **result}, ensure_ascii=False), flush=True)
    return result
