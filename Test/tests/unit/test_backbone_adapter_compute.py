"""CPU contracts for the DTK frozen-text backbone adapter recipe.

Real operator and optimizer assertions do not establish DTK hardware determinism.
"""

import copy
import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lokr import LoKr
from ypuddin.adapters.lora import LoRA
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    DTK_ANIMA_LORA_FSDP_PREVIEW_POLICY_ID,
    DTK_BACKBONE_ADAPTER_POLICY_IDS,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.train.trainer import Trainer


def config(family="anima", algo="lora", strategy="ddp"):
    return TrainConfig.model_validate(
        {
            "model": {"family": family, "attention": "flash_attn"},
            "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
            "adapter": {"algo": algo, "param_dtype": "fp32", "mode": "bypass"},
            "loop": {
                "gpu_count": 1 if strategy == "single" else 2,
                "distributed_strategy": "ddp" if strategy == "single" else strategy,
                "mixed_precision": "bf16",
                "deterministic": True,
            },
            "memory": {"allow_tf32": True},
        }
    )


@pytest.mark.parametrize("family,algo,strategy", list(DTK_BACKBONE_ADAPTER_POLICY_IDS))
def test_versioned_scope_and_strict_resume_identity(family, algo, strategy):
    cfg = config(family, algo, strategy)
    original = cfg.to_dict()
    effective, policy = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert cfg.to_dict() == original
    assert effective.loop.mixed_precision == "bf16"
    assert not effective.memory.allow_tf32 and effective.model.attention == "sdpa"
    assert policy["id"] == (
        DTK_ANIMA_LORA_FSDP_PREVIEW_POLICY_ID
        if (family, algo, strategy) == ("anima", "lora", "fsdp")
        else DTK_BACKBONE_ADAPTER_POLICY_IDS[(family, algo, strategy)]
    )
    assert policy["operator_components"] == policy["trainable_components"] == ["backbone"]
    assert policy["distributed_strategy"] == strategy
    assert ("adapter_implementation" in policy) == (algo == "lora")
    assert ("fsdp_param_dtype" in policy) == (strategy == "fsdp")
    assert resolve_training_compute_config(effective, "cuda", "linux-dtk")[1] == policy
    validate_resume_compute_policy(policy, dict(policy))
    for saved in (
        None,
        policy | {"id": "old"},
        policy | {"adapter_algorithm": "other"},
        policy | {"linear_backward_implementation": "old"},
        policy | {"distributed_strategy": "other"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(policy, saved)
    for device, profile, deterministic in (
        ("cpu", "linux-dtk", True),
        ("cuda", "linux-cuda", True),
        ("cuda", "linux-dtk", False),
    ):
        cfg.loop.deterministic = deterministic
        unchanged, absent = resolve_training_compute_config(cfg, device, profile)
        assert absent is None and unchanged.to_dict() == cfg.to_dict()


def trainer(monkeypatch, *, algo="lora", strategy="ddp", dropout=0.0, rank_dropout=0.0):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    obj = object.__new__(Trainer)
    obj.cfg, obj.compute_policy = resolve_training_compute_config(
        config(algo=algo, strategy=strategy), "cuda", "linux-dtk"
    )
    obj.device = torch.device("cuda")  # Metadata only: all actual tensor math is CPU.
    adapter = (LoRA if algo == "lora" else LoKr)(
        8, 8, rank=2, alpha=2, dropout=dropout, rank_dropout=rank_dropout
    )
    for param in adapter.parameters():
        nn.init.normal_(param, std=0.2)
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), adapter, mode="bypass")
    obj.loaded = SimpleNamespace(backbone=layer)
    return obj


class Contractions(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}:
            self.dtypes.extend(arg.dtype for arg in args if isinstance(arg, torch.Tensor))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
@pytest.mark.parametrize("recompute", [False, True])
def test_backbone_lora_cold_optimizer_resume_and_real_fp32_contractions(monkeypatch, strategy, recompute):
    torch.manual_seed(2145)
    hot = trainer(monkeypatch, strategy=strategy, dropout=0.25, rank_dropout=0.25)
    cold = trainer(monkeypatch, strategy=strategy, dropout=0.25, rank_dropout=0.25)
    hot._install_backbone_adapter_compute_operators()
    optimizer = torch.optim.AdamW(hot.loaded.backbone.adapter.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8).transpose(0, 1).bfloat16()

    def update(obj, opt):
        obj._validate_training_compute_policy()
        opt.zero_grad(set_to_none=True)
        trace = Contractions()
        with torch.autocast("cpu", dtype=torch.bfloat16), trace:
            fn = obj.loaded.backbone
            y = checkpoint(fn, x, use_reentrant=False) if recompute else fn(x)
            loss = y.float().square().mean()
        with trace:
            loss.backward()
        assert trace.dtypes and set(trace.dtypes) == {torch.float32}
        assert all(
            p.grad is not None and p.grad.dtype == torch.float32
            for p in obj.loaded.backbone.adapter.parameters()
        )
        opt.step()
        return loss.detach().clone()

    update(hot, optimizer)
    buffer = io.BytesIO()
    torch.save(
        {
            "weights": hot.loaded.backbone.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "policy": hot.compute_policy,
        },
        buffer,
    )
    expected_loss = update(hot, optimizer)
    expected_rng = torch.get_rng_state().clone()
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    validate_resume_compute_policy(cold.compute_policy, saved["policy"])
    with pytest.raises(ValueError, match="未完整安装"):
        cold._validate_training_compute_policy()
    cold.loaded.backbone.load_state_dict(saved["weights"])
    cold._install_backbone_adapter_compute_operators()
    cold_opt = torch.optim.AdamW(cold.loaded.backbone.adapter.parameters(), lr=1e-3)
    cold_opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    assert torch.equal(update(cold, cold_opt), expected_loss)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in hot.loaded.backbone.state_dict().items():
        assert torch.equal(value, cold.loaded.backbone.state_dict()[name]), name
    for index, state in optimizer.state_dict()["state"].items():
        assert all(
            torch.equal(value, cold_opt.state_dict()["state"][index][key]) for key, value in state.items()
        )


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_fsdp_dynamic_adapter_class_keeps_validation_and_native_lokr(monkeypatch, algo):
    from torch.distributed.fsdp import FSDPModule

    obj = trainer(monkeypatch, algo=algo, strategy="fsdp")
    adapter = obj.loaded.backbone.adapter
    native = copy.deepcopy(adapter)
    obj._install_backbone_adapter_compute_operators()
    # fully_shard applies this same dynamic inheritance; no fake sharding claims.
    adapter.__class__ = type(f"FSDP{type(adapter).__name__}", (FSDPModule, type(adapter)), {})
    obj._validate_training_compute_policy()
    if algo == "lokr":
        assert "delta_apply" not in adapter.__dict__
        x = torch.randn(2, 3, 8).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected, actual = native.delta_apply(x), adapter.delta_apply(x)
        assert torch.equal(expected, actual)
        expected.float().sum().backward()
        actual.float().sum().backward()
        assert all(
            torch.equal(p.grad, q.grad)
            for p, q in zip(native.parameters(), adapter.parameters(), strict=True)
        )
    else:
        adapter.delta_apply = lambda x: x
        with pytest.raises(ValueError, match="缺失或被替换"):
            obj._validate_training_compute_policy()


def test_fsdp_contract_cannot_be_installed_by_unsharded_trainer(monkeypatch):
    obj = trainer(monkeypatch, strategy="fsdp")
    with pytest.raises(ValueError, match="FSDP"):
        obj._place_training_model()


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_arbitrary_algorithm_subclasses_cannot_claim_native_compute_contract(monkeypatch, algo):
    obj = trainer(monkeypatch, algo=algo, strategy="fsdp")
    adapter = obj.loaded.backbone.adapter
    adapter.__class__ = type("ChangedAlgorithm", (type(adapter),), {})
    with pytest.raises(ValueError, match="实际"):
        obj._install_backbone_adapter_compute_operators()
