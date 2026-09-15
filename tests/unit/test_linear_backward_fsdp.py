"""Real two-rank CPU FSDP2, mixed precision, recomputation and exact resume."""

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed import fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint
from transformers.optimization import Adafactor

from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
    DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
    DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID,
    resolve_training_compute_config,
)
from ypuddin.optim.sharded import prepare_sharded_optimizer
from ypuddin.train.conv_forward import install_conv_fp32_forward, validate_conv_forward_installation
from ypuddin.train.linear_backward import (
    install_linear_bf16_forward_fp32_backward,
    validate_linear_backward_installation,
)
from ypuddin.train.sharded_state import load_sharded_checkpoint, save_sharded_checkpoint
from ypuddin.train.state import Progress


class Model(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.blocks = nn.ModuleList(
            [nn.Linear(17, 13), nn.Linear(13, 7)]
            if family == "krea2"
            else [nn.Conv2d(3, 4, 3, padding=1), nn.Linear(4, 7)]
        )

    def forward(self, x):
        for index, block in enumerate(self.blocks):
            if self.family == "sdxl" and index == 1:
                x = x.permute(0, 2, 3, 1)
            x = torch.tanh(checkpoint(block, x, use_reentrant=False))
        return x


class ConvBoundary(TorchDispatchMode):
    """Observe the actual convolution operands, not just configured precision."""

    def __init__(self):
        super().__init__()
        self.forward = []
        self.backward = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.convolution.default:
            self.forward.append(([value.dtype for value in args[:3] if value is not None], result.dtype))
        elif func is torch.ops.aten.convolution_backward.default:
            self.backward.append([value.dtype for value in args[:3]])
        return result


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return (value.to_local() if isinstance(value, DTensor) else value).detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    return value


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def worker(rank, directory, family):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(path / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        # Older supported Torch releases have FSDP2 but no CPU execution. Only
        # that capability failure may skip; operator/checkpoint failures cannot.
        probe = nn.Linear(4, 3)
        try:
            fsdp.fully_shard(probe, mesh=mesh)
            probe(torch.ones(2, 4)).sum().backward()
        except RuntimeError as error:
            reason = str(error)
            if any(
                text in reason.lower()
                for text in (
                    "requires cuda",
                    "does not support cpu",
                    "cpu device is not supported",
                    "device type cpu is not supported",
                )
            ):
                (path / f"rank-{rank}.json").write_text(json.dumps({"skip": reason}))
                return
            raise
        torch.manual_seed(442)
        model = Model(family)
        original_ids = [id(parameter) for parameter in model.parameters()]
        restore, counts = install_linear_bf16_forward_fp32_backward(model)
        conv_restore, conv_counts = (
            install_conv_fp32_forward(model) if family == "sdxl" else (lambda: None, None)
        )
        assert original_ids == [id(parameter) for parameter in model.parameters()]
        gathered_dtypes, output_dtypes = [], []
        hooks = []
        for block in [*model.blocks, model]:
            fsdp.fully_shard(
                block,
                mesh=mesh,
                mp_policy=fsdp.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
                ),
                reshard_after_forward=True,
            )
        for block in model.blocks:
            hooks.append(
                block.register_forward_pre_hook(
                    lambda module, inputs: gathered_dtypes.append(module.weight.dtype)
                )
            )
            hooks.append(
                block.register_forward_hook(lambda module, inputs, output: output_dtypes.append(output.dtype))
            )
        validate_linear_backward_installation(model, counts)
        if conv_counts is not None:
            validate_conv_forward_installation(model, conv_counts)
        optimizer = prepare_sharded_optimizer(
            Adafactor(model.parameters(), lr=1e-5, relative_step=False, scale_parameter=False, beta1=None)
            if family == "krea2"
            else torch.optim.AdamW(model.parameters(), lr=1e-5)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)

        def state():
            return snapshot(
                {
                    "weights": dict(model.named_parameters()),
                    "optimizer": dict(optimizer.state),
                    "scheduler": scheduler.state_dict(),
                }
            )

        def step(index):
            generator = torch.Generator().manual_seed(442 + rank * 10 + index)
            input_shape, target_shape = (
                ((2, 5, 17), (2, 5, 7)) if family == "krea2" else ((2, 3, 5, 7), (2, 5, 7, 7))
            )
            x = torch.randn(input_shape, generator=generator).requires_grad_()
            target = torch.randn(target_shape, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            observed = ConvBoundary()
            with observed:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    prediction = model(x)
                    assert prediction.dtype == torch.bfloat16
                    loss = (prediction.float() - target).square().mean()
                loss.backward()
            assert gathered_dtypes and set(gathered_dtypes) == {torch.bfloat16}
            assert output_dtypes and set(output_dtypes) == {torch.bfloat16}
            if family == "sdxl":
                assert observed.forward and observed.backward
                assert all(
                    set(inputs) == {torch.float32} and output == torch.float32
                    for inputs, output in observed.forward
                )
                assert all(set(inputs) == {torch.float32} for inputs in observed.backward)
            else:
                assert not observed.forward and not observed.backward
            for parameter in model.parameters():
                assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32
                assert parameter.grad.dtype == parameter.grad.to_local().dtype == torch.float32
                assert torch.isfinite(parameter.grad.to_local()).all()
            optimizer.step()
            scheduler.step()
            return loss.detach().clone()

        initial = state()["weights"]
        for index in range(2):
            step(index)
        cfg = TrainConfig.model_validate(
            {
                "model": {"family": family},
                "training": {"mode": "full"},
                "loop": {
                    "gpu_count": 2,
                    "distributed_strategy": "fsdp",
                    "deterministic": True,
                    "mixed_precision": "bf16",
                },
            }
        )
        _, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
        assert policy["id"] == (
            DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID
            if family == "krea2"
            else DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID
        )
        assert policy["fsdp_param_dtype"] == "bfloat16" and policy["fsdp_reduce_dtype"] == "float32"
        runtime = {"torch": str(torch.__version__), "device_type": "cpu", "scope": "CPU contract only"}
        saved_path = path / "state-2"
        saved_rng = {"torch": torch.get_rng_state(), "rank": rank}
        save_sharded_checkpoint(
            saved_path,
            modules={"backbone": model},
            optimizer=optimizer,
            scheduler=scheduler,
            sampler_state={"rank": rank, "position": 2},
            progress=Progress(
                step=2, extra={"deterministic": True, "compute_policy": policy, "compute_runtime": runtime}
            ),
            rng=saved_rng,
            batch_size=1,
            grad_accum=1,
        )
        expected_loss = step(2)
        reference = state()
        assert not exact(initial, reference["weights"])
        checks = {
            "modules": {"backbone": model},
            "optimizer": optimizer,
            "expected_world_size": 2,
            "expected_deterministic": True,
            "expected_compute_policy": policy,
            "expected_compute_runtime": runtime,
        }
        # With a wrong policy on both ranks, reject metadata before even opening
        # model tensors or loading optimizer/RNG tensors. Asymmetric guards below
        # separately prove one rank's mismatch reaches peers before mutation.
        wrong_policy = policy | {"id": DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID}
        with (
            patch(
                "ypuddin.train.sharded_state.safe_open",
                side_effect=AssertionError("model opened before policy rejection"),
            ) as model_open,
            patch(
                "torch.load", side_effect=AssertionError("state loaded before policy rejection")
            ) as tensor_load,
        ):
            with pytest.raises(ValueError, match="计算"):
                load_sharded_checkpoint(saved_path, **(checks | {"expected_compute_policy": wrong_policy}))
            model_open.assert_not_called()
            tensor_load.assert_not_called()
        assert exact(reference, state())
        for override in (
            {
                "expected_compute_policy": policy | {"linear_backward_implementation": "old-candidate"}
                if rank
                else policy
            },
            {"expected_compute_runtime": runtime | {"torch": "changed"} if rank else runtime},
        ):
            with pytest.raises(ValueError, match="计算"):
                load_sharded_checkpoint(saved_path, **(checks | override))
            assert exact(reference, state())
        optimizer.state.clear()
        ids = [id(parameter) for parameter in model.parameters()]
        loaded = load_sharded_checkpoint(saved_path, **checks)
        assert ids == [id(parameter) for parameter in model.parameters()]
        assert loaded["sampler"] == {"rank": rank, "position": 2}
        assert loaded["progress"].step == 2
        assert loaded["progress"].extra["compute_policy"] == policy
        assert exact(loaded["rng"]["distributed"]["ranks"][rank], saved_rng)
        scheduler.load_state_dict(loaded["scheduler"])
        assert torch.equal(expected_loss, step(2))
        assert exact(reference, state())
        validate_linear_backward_installation(model, counts)
        if conv_counts is not None:
            validate_conv_forward_installation(model, conv_counts)
        conv_restore()
        restore()
        for hook in hooks:
            hook.remove()
        (path / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "family": family,
                    "optimizer": type(optimizer).__name__,
                    "policy_id": policy["id"],
                    "actual_all_gather_dtype": "bfloat16",
                    "actual_gradient_dtype": "float32",
                    "actual_conv_fp32": family == "sdxl",
                    "torch": str(torch.__version__),
                    "scope": "Actual two-rank CPU FSDP2, not DTK/GPU acceptance",
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["krea2", "sdxl"])
def test_two_rank_bf16_forward_fp32_backward_and_exact_sharded_resume(tmp_path, family):
    if not hasattr(fsdp, "fully_shard"):
        pytest.skip("Installed Torch lacks FSDP2")
    context = mp.spawn(worker, args=(str(tmp_path), family), nprocs=2, join=False)
    try:
        assert context.join(timeout=60) or context.join(timeout=10), "FSDP workers timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    if any("skip" in result for result in results):
        pytest.skip("; ".join(result.get("skip", "") for result in results))
    assert all(result["passed"] for result in results)
