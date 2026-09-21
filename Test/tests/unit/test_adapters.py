import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ypuddin.adapters import (
    AdaptedLinear,
    FrozenLinear,
    Full,
    LoHa,
    LoKr,
    LoRA,
    TargetPreset,
    build_metadata,
    factorization,
    inject,
    load_adapter_file,
    match_name,
    modules_from_tensors,
    resolve_targets,
    save_adapter_file,
)
from ypuddin.adapters.convert import comfy_to_kohya, kohya_to_comfy
from ypuddin.config import AdapterConfig, AdapterRule

torch.manual_seed(0)


# --------------------------------------------------------------------------- factorization
@pytest.mark.parametrize(
    "dim,factor,expected",
    [
        (1024, -1, (32, 32)),
        (1024, 4, (4, 256)),
        (1024, 8, (8, 128)),
        (1024, 16, (16, 64)),
        (1024, 64, (16, 64)),
        (1536, -1, (32, 48)),
        (2048, -1, (32, 64)),
        (3072, -1, (48, 64)),
        (3072, 8, (8, 384)),
        (3072, 10, (8, 384)),
        (4096, -1, (64, 64)),
        (8192, -1, (64, 128)),
        (12288, -1, (96, 128)),
        (360, 16, (15, 24)),
        (127, -1, (1, 127)),
    ],
)
def test_factorization_matches_lycoris(dim, factor, expected):
    assert factorization(dim, factor) == expected


# --------------------------------------------------------------------------- helpers
def _randomize(mod: nn.Module) -> None:
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn_like(p) * 0.1)


def _kron_reference(lokr: LoKr) -> torch.Tensor:
    w1 = lokr.w1_a @ lokr.w1_b if lokr.w1_lowrank else lokr.w1
    w2 = lokr.w2_a @ lokr.w2_b if lokr.w2_lowrank else lokr.w2
    scalar = lokr.scalar if lokr.scalar is not None else 1.0
    return torch.kron(w1, w2) * lokr.scale * scalar


LOKR_CASES = [
    dict(out=64, inp=48, rank="full", alpha=1.0, factor=-1),  # both full, scale 1
    dict(out=64, inp=48, rank=4, alpha=1.0, factor=-1),  # low-rank W2, alpha != rank
    dict(out=96, inp=64, rank=4, alpha=2.0, factor=8, decompose_both=True),  # both low-rank
    dict(out=64, inp=64, rank=4, alpha=4.0, factor=4, rs_lora=True),
    dict(out=48, inp=96, rank=2, alpha=1.0, factor=-1, init="scalar"),
]


@pytest.mark.parametrize("case", LOKR_CASES)
def test_lokr_delta_equals_kron(case):
    out, inp = case.pop("out"), case.pop("inp")
    lokr = LoKr(out, inp, **case)
    _randomize(lokr)
    torch.testing.assert_close(lokr.delta_weight(), _kron_reference(lokr), rtol=1e-5, atol=1e-6)
    assert lokr.delta_weight().shape == (out, inp)


@pytest.mark.parametrize("case", copy.deepcopy(LOKR_CASES))
def test_lokr_bypass_equals_linear_with_delta(case):
    out, inp = case.pop("out"), case.pop("inp")
    lokr = LoKr(out, inp, **case).eval()
    _randomize(lokr)
    x = torch.randn(3, 5, inp)
    torch.testing.assert_close(lokr.delta_apply(x), F.linear(x, lokr.delta_weight()), rtol=1e-5, atol=1e-5)


def test_lokr_shapes_and_param_count_3072():
    full = LoKr(3072, 3072, rank="full", factor=-1)
    assert (full.a, full.b, full.c, full.d) == (48, 64, 48, 64)
    assert full.num_params() == 48 * 48 + 64 * 64 == 6400
    assert full.scale == 1.0
    lr = LoKr(3072, 3072, rank=16, alpha=1.0, factor=8)
    assert lr.w1.shape == (8, 8) and lr.w2_a.shape == (384, 16) and lr.w2_b.shape == (16, 384)
    assert lr.num_params() == 64 + 6144 + 6144
    assert lr.scale == pytest.approx(1 / 16)
    rect = LoKr(12288, 3072, rank="full", factor=-1)
    assert rect.w1.shape == (96, 48) and rect.w2.shape == (128, 64)


@pytest.mark.parametrize("algo", ["lora", "lokr", "loha", "full"])
def test_initial_delta_is_zero(algo):
    base = nn.Linear(40, 32)
    mod = {"lora": LoRA, "lokr": LoKr, "loha": LoHa, "full": Full}[algo](
        32, 40, **({} if algo == "full" else {"rank": 4})
    )
    layer = AdaptedLinear(FrozenLinear.from_linear(base), mod, name="t")
    x = torch.randn(2, 40)
    torch.testing.assert_close(layer(x), base(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("algo", ["lora", "lokr", "loha"])
def test_scalar_init_delta_is_zero_and_trainable(algo):
    base = nn.Linear(40, 32)
    mod = {"lora": LoRA, "lokr": LoKr, "loha": LoHa}[algo](32, 40, rank=4, init="scalar")
    layer = AdaptedLinear(FrozenLinear.from_linear(base), mod, mode="auto")
    x = torch.randn(2, 40)
    torch.testing.assert_close(layer(x), base(x), rtol=1e-5, atol=1e-6)
    layer(x).sum().backward()
    assert mod.scalar.grad is not None and mod.scalar.grad.abs().sum() > 0


# --------------------------------------------------------------------------- mode equivalence
@pytest.mark.parametrize(
    "algo,kwargs",
    [
        ("lora", dict(rank=4, alpha=1.0)),
        ("lora", dict(rank=4, alpha=8.0, rs_lora=True)),
        ("lokr", dict(rank=4, alpha=1.0, factor=-1)),
        ("lokr", dict(rank="full", alpha=1.0, factor=4)),
        ("lokr", dict(rank=2, alpha=3.0, factor=8, decompose_both=True, init="scalar")),
    ],
)
@pytest.mark.parametrize("multiplier", [1.0, 0.5, 0.0])
def test_bypass_equals_merged(algo, kwargs, multiplier):
    base = nn.Linear(64, 48, bias=True)
    cls = {"lora": LoRA, "lokr": LoKr}[algo]
    adapter = cls(48, 64, **kwargs)
    _randomize(adapter)
    frozen = FrozenLinear.from_linear(base)
    bypass = AdaptedLinear(frozen, adapter, mode="bypass").eval()
    merged = AdaptedLinear(frozen, adapter, mode="merged").eval()
    bypass.multiplier = merged.multiplier = multiplier
    x = torch.randn(2, 7, 64)
    ref = base(x) + multiplier * F.linear(x, adapter.delta_weight())
    torch.testing.assert_close(bypass(x), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(merged(x), ref, rtol=1e-5, atol=1e-5)


def test_loha_delta_and_grad():
    loha = LoHa(32, 24, rank=4, alpha=2.0)
    _randomize(loha)
    ref = ((loha.w1_a @ loha.w1_b) * (loha.w2_a @ loha.w2_b)) * loha.scale
    torch.testing.assert_close(loha.delta_weight(), ref)
    loha.delta_weight().square().sum().backward()
    grads = [p.grad for p in loha.parameters()]
    assert all(g is not None for g in grads)
    # compare against autograd on the plain expression
    loha2 = copy.deepcopy(loha)
    for p in loha2.parameters():
        p.grad = None
    (((loha2.w1_a @ loha2.w1_b) * (loha2.w2_a @ loha2.w2_b)) * loha2.scale).square().sum().backward()
    for p1, p2 in zip(loha.parameters(), loha2.parameters(), strict=True):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)


def test_dora_initial_identity_and_rescale():
    base = nn.Linear(32, 16)
    adapter = LoRA(16, 32, rank=4)
    layer = AdaptedLinear(FrozenLinear.from_linear(base), adapter, dora=True)
    assert layer.mode == "merged"
    torch.testing.assert_close(layer.dora.dora_scale.squeeze(), base.weight.norm(dim=1), rtol=1e-6, atol=1e-6)
    x = torch.randn(3, 32)
    torch.testing.assert_close(layer(x), base(x), rtol=1e-5, atol=1e-5)
    _randomize(adapter)
    w = layer.merged_weight()
    torch.testing.assert_close(w.norm(dim=1), layer.dora.dora_scale.squeeze(), rtol=1e-5, atol=1e-5)


def test_rank_dropout_and_dropout_only_in_training():
    lokr = LoKr(32, 32, rank=4, alpha=4.0, rank_dropout=0.5, dropout=0.5)
    _randomize(lokr)
    x = torch.randn(4, 32)
    lokr.eval()
    torch.testing.assert_close(lokr.delta_apply(x), F.linear(x, lokr.delta_weight()))
    lokr.train()
    torch.manual_seed(1)
    a = lokr.delta_apply(x)
    torch.manual_seed(2)
    b = lokr.delta_apply(x)
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- fp8 base
def test_fp8_frozen_linear_dequant_close():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no fp8 dtype")
    w = torch.randn(64, 48) * 0.02
    fz = FrozenLinear(w, None, precision="fp8_e4m3")
    assert fz.is_fp8 and fz.weight.dtype == torch.float8_e4m3fn
    deq = fz.dequant(torch.float32)
    rel = (deq - w).norm() / w.norm()
    assert rel < 0.08
    x = torch.randn(2, 48)
    torch.testing.assert_close(fz(x), F.linear(x, deq), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- rules
@pytest.mark.parametrize(
    "pattern,name,expected",
    [
        ("blocks.*.self_attn.{q,k,v}_proj", "blocks.0.self_attn.k_proj", True),
        ("blocks.*.self_attn.{q,k,v}_proj", "blocks.0.self_attn.output_proj", False),
        ("re:^blocks\\.(0|1)\\..*", "blocks.1.mlp.layer1", True),
        ("re:^blocks\\.(0|1)\\..*", "blocks.2.mlp.layer1", False),
        ("*mlp*", "blocks.3.mlp.layer2", True),
    ],
)
def test_match_name(pattern, name, expected):
    assert match_name(pattern, name) is expected


def _preset() -> TargetPreset:
    return TargetPreset(
        "attn-mlp", include=("blocks.*.attn.{q,k,v,o}", "blocks.*.mlp.*"), exclude=("*adaln*",)
    )


def test_resolve_targets_precedence():
    names = ["blocks.0.attn.q", "blocks.0.attn.o", "blocks.0.mlp.fc1", "blocks.0.adaln.lin", "embed.proj"]
    cfg = AdapterConfig(
        algo="lokr",
        rank=8,
        rules=[
            AdapterRule(match="blocks.*.mlp.*", rank=4, alpha=2.0),
            AdapterRule(match="blocks.*.attn.o", algo="none"),
            AdapterRule(match="embed.*", algo="lora", rank=2),
        ],
    )
    targets = {t.name: t for t in resolve_targets(names, cfg, _preset())}
    assert set(targets) == {"blocks.0.attn.q", "blocks.0.mlp.fc1", "embed.proj"}
    assert (
        targets["blocks.0.mlp.fc1"].params["rank"] == 4 and targets["blocks.0.mlp.fc1"].params["alpha"] == 2.0
    )
    assert targets["blocks.0.attn.q"].params["rank"] == 8
    assert targets["embed.proj"].algo == "lora" and targets["embed.proj"].params["rank"] == 2


# --------------------------------------------------------------------------- injection
class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict(
            {"q": nn.Linear(32, 32), "k": nn.Linear(32, 32), "v": nn.Linear(32, 32), "o": nn.Linear(32, 32)}
        )
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.adaln = nn.ModuleDict({"lin": nn.Linear(32, 96)})

    def forward(self, x):
        h = self.attn["o"](self.attn["q"](x) + self.attn["k"](x) + self.attn["v"](x))
        return x + h + self.mlp(x) + self.adaln["lin"](x)[..., :32]


class _Toy(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.embed = nn.Linear(16, 32)
        self.blocks = nn.ModuleList([_Block() for _ in range(n)])
        self.head = nn.Linear(32, 16)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def test_inject_eject_roundtrip():
    model = _Toy()
    sd_before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(2, 16)
    y0 = model(x)
    preset = TargetPreset("attn-mlp", include=("blocks.*.attn.*", "blocks.*.mlp.*"))
    cfg = AdapterConfig(algo="lokr", rank=4, alpha=4.0, factor=-1)
    aset = inject(model, cfg, preset)
    assert len(aset.layers) == 2 * (4 + 2)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".adapter." in n for n in trainable)
    torch.testing.assert_close(model(x), y0, rtol=1e-5, atol=1e-5)  # zero init
    groups = aset.param_groups(1e-3, 0.01)
    assert any(g["weight_decay"] == 0.0 and g["name"] == "w1" for g in groups)
    aset.eject()
    assert {k: v for k, v in model.state_dict().items()}.keys() == sd_before.keys()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, sd_before[k])


def test_save_load_roundtrip_and_third_party_alpha_convention(tmp_path):
    model = _Toy()
    preset = TargetPreset("attn-mlp", include=("blocks.*.attn.*", "blocks.*.mlp.*"))
    cfg = AdapterConfig(
        algo="lokr",
        rank=4,
        alpha=1.0,
        factor=-1,
        rules=[AdapterRule(match="blocks.*.mlp.*", algo="lora", rank=2, alpha=1.0)],
    )
    aset = inject(model, cfg, preset)
    for layer in aset.layers.values():
        _randomize(layer.adapter)
    x = torch.randn(2, 16)
    y_ref = model(x)
    tensors, targets = aset.export_state()
    meta = build_metadata(
        targets=targets,
        adapter_cfg=cfg.model_dump(mode="json"),
        family="toy",
        architecture="toy/lora",
        title="t",
    )
    path = save_adapter_file(tmp_path / "a.safetensors", tensors, meta, dtype="fp32")
    loaded, meta2 = load_adapter_file(path)
    assert meta2["ypuddin.family"] == "toy" and "modelspec.hash_sha256" in meta2

    # 1) reload into a fresh model that shares the same base weights
    aset.eject()
    model2 = _Toy()
    model2.load_state_dict(model.state_dict())
    aset2 = inject(model2, cfg, preset)
    aset2.load_state(loaded)
    torch.testing.assert_close(model2(x), y_ref, rtol=1e-5, atol=1e-5)

    # 2) third-party loader semantics: scale = alpha / rank (rank from w2_b / down rows); full -> scale 1
    mods = modules_from_tensors(loaded, meta2)
    for key, (mod, _) in mods.items():
        sub = {k.split(".", 1)[1]: v for k, v in loaded.items() if k.startswith(key + ".")}
        if isinstance(mod, LoKr):
            w1 = sub["lokr_w1"]
            if "lokr_w2_a" in sub:
                rank = sub["lokr_w2_b"].shape[0]
                delta = torch.kron(w1, sub["lokr_w2_a"] @ sub["lokr_w2_b"]) * (sub["alpha"].item() / rank)
            else:
                delta = torch.kron(w1, sub["lokr_w2"])
        else:
            down, up = sub["lora_down.weight"], sub["lora_up.weight"]
            delta = (up @ down) * (sub["alpha"].item() / down.shape[0])
        torch.testing.assert_close(mod.delta_weight(), delta, rtol=1e-5, atol=1e-6)


def test_convert_kohya_comfy_roundtrip():
    names = ["blocks.0.self_attn.q_proj", "blocks.0.mlp.layer1"]
    tensors = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.zeros(4, 8),
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.zeros(8, 4),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
        "lora_unet_blocks_0_mlp_layer1.lokr_w1": torch.zeros(2, 2),
        "lora_unet_blocks_0_mlp_layer1.lokr_w2": torch.zeros(4, 4),
        "lora_unet_blocks_0_mlp_layer1.alpha": torch.tensor(1.0),
    }
    comfy = kohya_to_comfy(tensors, names)
    assert "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight" in comfy
    assert "lora_unet_blocks_0_mlp_layer1.lokr_w1" in comfy  # LoKr keys untouched
    back = comfy_to_kohya(comfy)
    assert set(back) == set(tensors)


def test_full_adapter_trains_and_exports_diff():
    base = nn.Linear(8, 8)
    full = Full(8, 8)
    layer = AdaptedLinear(FrozenLinear.from_linear(base), full)
    x = torch.randn(4, 8)
    torch.testing.assert_close(layer(x), base(x), rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        full.weight.add_(0.5)
    diff = full.export_tensors()["diff"]
    torch.testing.assert_close(diff, torch.full((8, 8), 0.5), rtol=1e-5, atol=1e-6)
    rebuilt = Full.from_tensors({"diff": diff})
    rebuilt.bind_base(base.weight.data)
    torch.testing.assert_close(rebuilt.weight, full.weight)
