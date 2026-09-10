"""Compatibility check against the real LyCORIS package (optional).

Runs only when ``lycoris`` is importable (``pip install lycoris-lora`` or
``pip install ../LyCORIS``); skipped otherwise. Verifies that LyCORIS's own
kohya loader reads a LoKr file written by ypuddin — with low-rank ``w2`` and
``alpha != rank``, the case where scale handling matters — and produces an
identical model output.
"""
import torch
from torch import nn

import pytest

lycoris = pytest.importorskip("lycoris", reason="lycoris-lora not installed")
import lycoris.kohya  # noqa: E402

from ypuddin.adapters import (  # noqa: E402
    TargetPreset,
    build_metadata,
    inject,
    save_adapter_file,
)
from ypuddin.config import AdapterConfig  # noqa: E402


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict({"q": nn.Linear(128, 128), "o": nn.Linear(128, 128)})
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, x):
        h = self.attn["o"](self.attn["q"](x))
        return x + h + self.mlp(x)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def test_lycoris_loads_ypuddin_lokr_file(tmp_path):
    torch.manual_seed(0)
    base = _Toy().eval()
    x = torch.randn(3, 128)

    # rank=4, factor=-1 on 128/256 dims -> low-rank w2 everywhere; alpha=2 != rank -> scale=0.5
    preset = TargetPreset("all", include=("blocks.*.attn.*", "blocks.*.mlp.*"))
    cfg = AdapterConfig(algo="lokr", rank=4, alpha=2.0, factor=-1)
    aset = inject(base, cfg, preset)
    with torch.no_grad():
        for layer in aset.layers.values():
            for p in layer.adapter.parameters():
                p.copy_(torch.randn_like(p) * 0.05)
    assert all(l.adapter.w2_lowrank for l in aset.layers.values()), "test must exercise the alpha path"
    y_ours = base(x)
    n_layers = len(aset.layers)
    tensors, targets = aset.export_state()
    meta = build_metadata(
        targets=targets,
        adapter_cfg=cfg.model_dump(mode="json"),
        family="toy",
        architecture="toy/lokr",
        title="lycoris-compat",
    )
    path = save_adapter_file(tmp_path / "a.safetensors", tensors, meta, dtype="fp32")
    aset.eject()

    model2 = _Toy().eval()
    model2.load_state_dict(base.state_dict())
    net, _ = lycoris.kohya.create_network_from_weights(1.0, str(path), None, None, model2)
    assert len(net.unet_loras) == n_layers
    assert {type(m).__name__ for m in net.unet_loras} == {"LokrModule"}

    net.apply_to(None, model2, False, True)
    y_theirs = model2(x)
    torch.testing.assert_close(y_theirs, y_ours, rtol=1e-5, atol=1e-5)
