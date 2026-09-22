"""Preview-only DTK policy boundaries and real CPU BF16 contraction coverage.

Hardware reproducibility must still pass the separate official-weight GPU runs.
"""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lokr import LoKr
from ypuddin.adapters.lora import LoRA
from ypuddin.config import TrainConfig
from ypuddin.config.compute_policy import (
    BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    BF16_LORA_FP32_IMPLEMENTATION_ID,
    DTK_ANIMA_FSDP_PREVIEW_POLICY_IDS,
    DTK_ANIMA_LORA_FSDP_PREVIEW_POLICY_ID,
    DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS,
    DTK_BACKBONE_ADAPTER_POLICY_IDS,
    DTK_SDXL_LONG_TEXT_POLICY_ID,
    DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS,
    resolve_training_compute_config,
    validate_resume_compute_policy,
)
from ypuddin.config.schema import AdapterRule
from ypuddin.train.linear_backward import install_linear_bf16_operands_fp32_compute
from ypuddin.train.preview_compute import preview_linear_compute
from ypuddin.train.trainer import Trainer


def config(algo="lora", strategy="ddp", length=150, family="sdxl"):
    return TrainConfig.model_validate(
        {
            "model": {"family": family, "sdxl_max_token_length": length},
            "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
            "adapter": {"algo": algo, "mode": "bypass", "param_dtype": "fp32"},
            "loop": {
                "gpu_count": 2,
                "distributed_strategy": strategy,
                "mixed_precision": "bf16",
                "deterministic": True,
            },
        }
    )


def policy(algo="lora", strategy="ddp", length=150):
    return resolve_training_compute_config(config(algo, strategy, length), "cuda", "linux-dtk")[1]


@pytest.mark.parametrize("gpu_count", [2, 4])
@pytest.mark.parametrize("adapter_mode", ["auto", "bypass"])
@pytest.mark.parametrize("checkpointing", ["none", "block"])
def test_anima_fsdp_preview_keeps_legacy_training_fields_and_versions_resume(
    gpu_count, adapter_mode, checkpointing
):
    cfg = config(strategy="fsdp", family="anima")
    cfg.loop.gpu_count = gpu_count
    cfg.adapter.mode = adapter_mode
    cfg.memory.activation_checkpointing = checkpointing
    original = cfg.to_dict()
    effective, actual = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    # Snapshot the previous v1 contract, including every training dtype/operator.
    legacy = {
        "id": DTK_BACKBONE_ADAPTER_POLICY_IDS[("anima", "lora", "fsdp")],
        "mixed_precision": "bf16",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
        "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "linear_backward": "fp32-contractions-grad-original-dtype",
        "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        "adapter_algorithm": "lora",
        "trainable_components": ["backbone"],
        "operator_components": ["backbone"],
        "distributed_strategy": "fsdp",
        "adapter_forward": "bf16-rounded-operands-fp32-contractions-bf16-intermediates",
        "adapter_backward": "fp32-contractions-grad-original-dtype",
        "adapter_implementation": BF16_LORA_FP32_IMPLEMENTATION_ID,
        "fsdp_param_dtype": "bfloat16",
        "fsdp_reduce_dtype": "float32",
    }
    assert actual == legacy | {
        "id": DTK_ANIMA_LORA_FSDP_PREVIEW_POLICY_ID,
        "preview_operator_components": ["backbone"],
        "preview_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "preview_linear_implementation": BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    }
    assert actual["id"] in DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS
    assert cfg.to_dict() == original
    expected_effective = copy.deepcopy(original)
    expected_effective["model"]["attention"] = "sdpa"
    expected_effective["memory"]["allow_tf32"] = False
    assert effective.to_dict() == expected_effective
    assert resolve_training_compute_config(effective, "cuda", "linux-dtk")[1] == actual
    validate_resume_compute_policy(actual, copy.deepcopy(actual))
    # Neither direction may silently mix the old and new preview numeric recipe.
    with pytest.raises(ValueError, match="计算"):
        validate_resume_compute_policy(actual, legacy)
    with pytest.raises(ValueError, match="计算"):
        validate_resume_compute_policy(legacy, actual)
    for field in ("preview_operator_components", "preview_linear_forward", "preview_linear_implementation"):
        missing = {key: value for key, value in actual.items() if key != field}
        changed = actual | {field: ["text_encoder"] if field == "preview_operator_components" else "old"}
        for saved in (missing, changed):
            with pytest.raises(ValueError, match="计算"):
                validate_resume_compute_policy(actual, saved)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("loop", "gpu_count", 1),
        ("loop", "distributed_strategy", "ddp"),
        ("loop", "deterministic", False),
        ("loop", "mixed_precision", "fp16"),
        ("loop", "mixed_precision", "no"),
        ("training", "mode", "full"),
        ("training", "train_backbone", False),
        ("training", "train_text_encoder", True),
        ("adapter", "algo", "loha"),
        ("adapter", "mode", "merged"),
        ("adapter", "param_dtype", "bf16"),
        ("adapter", "dora", True),
        ("adapter", "rules", [AdapterRule(match="*", algo="lokr")]),
        ("memory", "base_precision", "fp8_e4m3"),
        ("memory", "activation_checkpointing", "unsloth"),
        ("memory", "compile", True),
        ("memory", "blocks_to_swap", 1),
    ],
)
def test_anima_preview_does_not_expand_existing_supported_recipe(section, field, value):
    cfg = config(strategy="fsdp", family="anima")
    setattr(getattr(cfg, section), field, value)
    _, actual = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert not any(key.startswith("preview_") for key in (actual or {}))


@pytest.mark.parametrize(
    "device,profile", [("cpu", "linux-dtk"), ("cuda", "linux-cuda"), ("cuda", "windows-cuda")]
)
def test_anima_preview_is_not_selected_outside_dtk_gpu(device, profile):
    cfg = config(strategy="fsdp", family="anima")
    effective, actual = resolve_training_compute_config(cfg, device, profile)
    assert actual is None and effective.to_dict() == cfg.to_dict()


@pytest.mark.parametrize("text_mode,optimizer", [("cached", "adamw"), ("online", "sgd")])
@pytest.mark.parametrize("checkpointing", ["none", "block"])
def test_anima_lokr_preview_only_versions_inference_with_online_dropout(text_mode, optimizer, checkpointing):
    cfg = config(algo="lokr", strategy="fsdp", family="anima")
    cfg.dataset.text_encoding = text_mode
    cfg.dataset.cache_latents = text_mode == "cached"
    cfg.memory.activation_checkpointing = checkpointing
    cfg.optimizer.type = optimizer
    cfg.optimizer.args = {"momentum": 0.9} if optimizer == "sgd" else {}
    cfg.adapter.dropout, cfg.adapter.rank_dropout = 0.1, 0.1
    _, actual = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    legacy = {
        "id": DTK_BACKBONE_ADAPTER_POLICY_IDS[("anima", "lokr", "fsdp")],
        "mixed_precision": "bf16", "allow_tf32": False, "attention": "sdpa", "sdpa_backend": "math",
        "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "linear_backward": "fp32-contractions-grad-original-dtype",
        "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        "adapter_algorithm": "lokr", "trainable_components": ["backbone"],
        "operator_components": ["backbone"], "distributed_strategy": "fsdp",
        "fsdp_param_dtype": "bfloat16", "fsdp_reduce_dtype": "float32",
    }
    assert actual == legacy | {
        "id": DTK_ANIMA_FSDP_PREVIEW_POLICY_IDS["lokr"],
        "preview_operator_components": ["backbone"],
        "preview_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "preview_linear_implementation": BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    }
    assert not any(key.startswith("adapter_") for key in actual if key != "adapter_algorithm")
    for expected, saved in ((actual, legacy), (legacy, actual)):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(expected, saved)
    for field in ("preview_operator_components", "preview_linear_forward", "preview_linear_implementation"):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(actual, {key: value for key, value in actual.items() if key != field})
    for section, field, value in (("loop", "gpu_count", 1), ("loop", "distributed_strategy", "ddp"),
                                  ("loop", "deterministic", False), ("training", "train_text_encoder", True)):
        changed = cfg.model_copy(deep=True)
        setattr(getattr(changed, section), field, value)
        _, excluded = resolve_training_compute_config(changed, "cuda", "linux-dtk")
        assert not any(key.startswith("preview_") for key in (excluded or {}))


def test_lokr_preview_context_preserves_dropout_training_and_native_eval_delta():
    torch.manual_seed(144)
    adapter = LoKr(8, 8, rank=2, dropout=0.3, rank_dropout=0.25)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_(std=0.1)
    first = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), adapter, mode="bypass")
    second = copy.deepcopy(first)
    restores = [install_linear_bf16_operands_fp32_compute(layer)[0] for layer in (first, second)]
    cfg = config(algo="lokr", strategy="fsdp", family="anima")
    selected = resolve_training_compute_config(cfg, "cuda", "linux-dtk")[1]
    x = torch.randn(2, 3, 8)
    try:
        outputs, grads, rngs = [], [], []
        for layer, context in ((first, nullcontext()), (second, preview_linear_compute(selected))):
            torch.manual_seed(385)
            layer.train()
            with context, torch.autocast("cpu", dtype=torch.bfloat16):
                output = layer(x)
            output.float().square().sum().backward()
            outputs.append(output.detach())
            grads.append([parameter.grad.clone() for parameter in layer.adapter.parameters()])
            rngs.append(torch.get_rng_state().clone())
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(rngs[0], rngs[1])
        assert all(torch.equal(left, right) for left, right in zip(*grads, strict=True))
        first.eval(), second.eval()
        trace = Contractions()
        before = torch.get_rng_state().clone()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = first.adapter.delta_apply(x.bfloat16())
            with preview_linear_compute(selected), trace:
                actual = second.adapter.delta_apply(x.bfloat16())
        assert torch.equal(actual, expected) and torch.equal(before, torch.get_rng_state())
        assert trace.dtypes and set(trace.dtypes) == {torch.bfloat16}
    finally:
        for restore in restores:
            restore()


@pytest.mark.parametrize("algo", ["lora", "lokr"])
@pytest.mark.parametrize("strategy", ["single", "ddp", "fsdp"])
@pytest.mark.parametrize("length", [150, 225])
def test_preview_recipe_versions_only_inference_and_refuses_older_state(algo, strategy, length):
    cfg = config(algo, "ddp" if strategy == "single" else strategy, length)
    if strategy == "single":
        cfg.loop.gpu_count = 1
    original = cfg.to_dict()
    effective, actual = resolve_training_compute_config(cfg, "cuda", "linux-dtk")
    assert cfg.to_dict() == original
    assert actual["id"] == DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS[(algo, strategy)]
    assert actual["preview_operator_components"] == ["backbone"]
    assert actual["preview_linear_forward"] == "bf16-rounded-operands-fp32-contraction-bf16-output"
    assert actual["preview_linear_implementation"] == BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID
    assert actual["linear_backward_implementation"] == BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
    assert actual["distributed_strategy"] == strategy
    assert actual["adapter_algorithm"] == algo
    assert ("adapter_implementation" in actual) == (algo == "lora")
    # The unsharded recipe already has exactly these training contractions.
    single = cfg.model_copy(deep=True)
    single.loop.gpu_count, single.loop.distributed_strategy = 1, "ddp"
    _, training = resolve_training_compute_config(single, "cuda", "linux-dtk")
    for key in (
        "linear_forward",
        "linear_backward",
        "linear_backward_implementation",
        "conv_forward",
        "conv_implementation",
    ):
        assert actual[key] == training[key]
    if strategy == "fsdp":
        assert actual["fsdp_param_dtype"] == "bfloat16" and actual["fsdp_reduce_dtype"] == "float32"
    assert resolve_training_compute_config(effective, "cuda", "linux-dtk")[1] == actual
    validate_resume_compute_policy(actual, dict(actual))
    old = {key: value for key, value in actual.items() if not key.startswith("preview_")}
    old["id"] = (
        DTK_SDXL_LONG_TEXT_POLICY_ID
        if algo == "lokr" and strategy in {"single", "ddp"}
        else DTK_BACKBONE_ADAPTER_POLICY_IDS[("sdxl", algo, strategy)]
    )
    for saved in (
        None,
        old,
        actual | {"preview_linear_implementation": "old"},
        actual | {"preview_operator_components": ["text_encoder"]},
        actual | {"preview_linear_forward": "native-bf16"},
    ):
        with pytest.raises(ValueError, match="计算"):
            validate_resume_compute_policy(actual, saved)


@pytest.mark.parametrize("change", ["75", "anima", "text", "not_deterministic", "nvidia"])
def test_preview_does_not_expand_to_unmeasured_or_unrelated_scope(change):
    cfg = config()
    profile = "linux-dtk"
    if change == "75":
        cfg.model.sdxl_max_token_length = 75
    elif change == "anima":
        cfg.model.family = "anima"
    elif change == "text":
        cfg.training.train_text_encoder = True
        cfg.dataset.text_encoding = "online"
    elif change == "not_deterministic":
        cfg.loop.deterministic = False
    elif change == "nvidia":
        profile = "linux-cuda"
    _, actual = resolve_training_compute_config(cfg, "cuda", profile)
    assert not any(key.startswith("preview_") for key in (actual or {}))


class Contractions(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}:
            self.dtypes.extend(arg.dtype for arg in args if isinstance(arg, torch.Tensor))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_preview_preserves_bf16_rounding_and_does_not_leak_to_normal_no_grad(frozen, bias, contiguous):
    torch.manual_seed(446)
    source = nn.Linear(7, 9, bias=bias)
    model = FrozenLinear.from_linear(source) if frozen else source
    native = copy.deepcopy(model)
    restore, _ = install_linear_bf16_operands_fp32_compute(model)
    x = torch.randn(3, 2, 7).transpose(0, 1)
    if contiguous:
        x = x.contiguous()
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected_native = native(x)
            assert torch.equal(model(x), expected_native)
            before_rng = torch.get_rng_state().clone()
            trace = Contractions()
            with preview_linear_compute(policy()), trace:
                actual = model(x)
            assert trace.dtypes and set(trace.dtypes) == {torch.float32}
            assert actual.dtype == torch.bfloat16
            assert torch.equal(torch.get_rng_state(), before_rng)
            assert torch.equal(model(x), expected_native)
            with preview_linear_compute(policy()):
                with preview_linear_compute(None):
                    trace = Contractions()
                    with trace:
                        disabled = model(x)
                    assert set(trace.dtypes) == {torch.bfloat16}
                    assert torch.equal(disabled, expected_native)
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            if contiguous:
                expected = torch.nn.functional.linear(
                    x.bfloat16().float(),
                    source.weight.bfloat16().float(),
                    source.bias.bfloat16().float() if bias else None,
                ).bfloat16()
            else:
                # Native noncontiguous Linear dispatches matmul then adds bias
                # after the BF16 contraction output boundary. Do not fuse this
                # into a different FP32-bias Linear recipe.
                expected = (x.bfloat16().float() @ source.weight.bfloat16().float().T).bfloat16()
                if bias:
                    expected = expected + source.bias.bfloat16()
        assert torch.equal(actual, expected)
    finally:
        restore()


def test_preview_context_exception_restores_native_and_does_not_change_training_gradients():
    torch.manual_seed(862)
    baseline, candidate = nn.Linear(7, 9), nn.Linear(7, 9)
    candidate.load_state_dict(baseline.state_dict())
    r1, _ = install_linear_bf16_operands_fp32_compute(baseline)
    r2, _ = install_linear_bf16_operands_fp32_compute(candidate)
    x = torch.randn(2, 3, 7)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            first = baseline(a)
            with preview_linear_compute(policy()):
                second = candidate(b)
        first.float().square().sum().backward()
        second.float().square().sum().backward()
        assert torch.equal(first, second) and torch.equal(a.grad, b.grad)
        assert all(
            torch.equal(p.grad, q.grad)
            for p, q in zip(baseline.parameters(), candidate.parameters(), strict=True)
        )
        with pytest.raises(RuntimeError, match="sampling failed"):
            with preview_linear_compute(policy()):
                raise RuntimeError("sampling failed")
        trace = Contractions()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), trace:
            candidate(x)
        assert set(trace.dtypes) == {torch.bfloat16}
    finally:
        r1()
        r2()


@pytest.mark.parametrize(
    "family,algo,strategy",
    [("sdxl", algo, strategy) for algo in ("lora", "lokr") for strategy in ("ddp", "fsdp")]
    + [("anima", algo, "fsdp") for algo in ("lora", "lokr")],
)
def test_actual_trainer_installs_preview_coverage_and_leaves_adapter_native(
    monkeypatch, family, algo, strategy
):
    import ypuddin.train.trainer as module

    monkeypatch.setattr(module, "current_profile", lambda: "linux-dtk")
    obj = object.__new__(Trainer)
    obj.cfg, obj.compute_policy = resolve_training_compute_config(
        config(algo, strategy, family=family), "cuda", "linux-dtk"
    )
    obj.device = torch.device("cuda")
    adapter = (LoRA if algo == "lora" else LoKr)(8, 8, rank=2)
    native = copy.deepcopy(adapter)
    layer = AdaptedLinear(FrozenLinear.from_linear(nn.Linear(8, 8)), adapter, mode="bypass")
    obj.loaded = SimpleNamespace(backbone=nn.ModuleDict({"linear": layer, "conv": nn.Conv2d(1, 1, 1)}))
    obj._install_backbone_adapter_compute_operators()
    obj._validate_training_compute_policy()
    x = torch.randn(2, 3, 8).bfloat16()
    trace = Contractions()
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16),
        preview_linear_compute(obj.compute_policy),
    ):
        with trace:
            value = adapter.delta_apply(x)
        assert torch.equal(value, native.delta_apply(x))
    assert trace.dtypes and set(trace.dtypes) == {torch.bfloat16}


@pytest.mark.parametrize("family,algo,strategy", [("sdxl", "lora", "ddp"), ("anima", "lora", "fsdp"), ("anima", "lokr", "fsdp")])
def test_product_sample_entry_enables_only_backbone_predictions(tmp_path, family, algo, strategy):
    cfg = config(family=family, algo=algo, strategy=strategy)
    cfg.sampling.prompts = [{"prompt": "one"}]
    # Validate the complete sampling config after supplying a simple prompt.
    cfg = TrainConfig.model_validate(cfg.to_dict())
    cfg.sampling.width = cfg.sampling.height = 64
    cfg.sampling.shift, cfg.sampling.cfg = 1.0, 1.0
    cfg.sampling.output_dir = str(tmp_path)
    modules = {name: nn.Linear(8, 8) for name in ("text", "backbone", "vae")}
    restores = [install_linear_bf16_operands_fp32_compute(model)[0] for model in modules.values()]
    traces = {}

    def compute(name, x):
        trace = Contractions()
        with torch.autocast("cpu", dtype=torch.bfloat16), trace:
            value = modules[name](x)
        traces[name] = set(trace.dtypes)
        return value

    def sample(_loaded, predict, shape, **options):
        return predict(torch.randn(shape, generator=options["generator"]), torch.zeros(1))

    fake = SimpleNamespace(
        cfg=cfg,
        compute_policy=resolve_training_compute_config(cfg, "cuda", "linux-dtk")[1],
        device=torch.device("cpu"),
        run_dir=tmp_path,
        progress=SimpleNamespace(step=8, extra={}),
        is_primary=True,
        _evaluation=lambda **_: nullcontext(),
        _autocast=lambda: torch.autocast("cpu", dtype=torch.bfloat16),
        _text_cond=lambda _: compute("text", torch.ones(1, 8)),
        emit=lambda *_args, **_kwargs: None,
        loaded=SimpleNamespace(
            dtype=torch.bfloat16,
            latent=SimpleNamespace(to=lambda _: None, decode=lambda x: compute("vae", x)[:, :3]),
        ),
        family=SimpleNamespace(
            spec=SimpleNamespace(latent=SimpleNamespace(stride=8, patch=1, align=8, channels=4)),
            sampling_defaults=lambda _: SimpleNamespace(steps=1, cfg=1.0, guidance=1.0),
            sample_latents=sample,
            forward=lambda _loaded, x, _t, _c, **_: compute("backbone", x),
        ),
    )
    try:
        paths = Trainer.sample_images(fake, "test")
        assert len(paths) == 1 and paths[0].is_file()
        assert traces == {"text": {torch.bfloat16}, "backbone": {torch.float32}, "vae": {torch.bfloat16}}
    finally:
        for restore in restores:
            restore()
