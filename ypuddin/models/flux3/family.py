"""FLUX 3 Image Klein Base text-to-image adapter training."""

from __future__ import annotations

import torch
from safetensors import SafetensorError
from torch import nn

from ypuddin.adapters import TargetPreset
from ypuddin.config import ModelConfig
from ypuddin.models.base import (
    LatentSpec,
    LoadedModel,
    MemoryLayout,
    ModelFamily,
    ModelSpec,
    SamplingDefaults,
    TextCond,
    TextSpec,
)
from ypuddin.models.registry import register

from . import loading
from .dit import DiT, Packed
from .latent import Flux3Latent
from .text import Flux3Text


def combine_conditions(conditions: list[TextCond]) -> TextCond:
    width = conditions[0]["embeds"].shape[-1]
    length = max(item["embeds"].shape[1] for item in conditions)
    batches = []
    for item in conditions:
        tensor = item["embeds"]
        padded = tensor.new_zeros(tensor.shape[0], length, width)
        padded[:, :tensor.shape[1]] = tensor
        batches.append(padded)
    return TextCond({"embeds": torch.cat(batches), "lengths": torch.cat([c["lengths"] for c in conditions])})


def pack_inputs(x_t, t, cond, dtype) -> Packed:
    batch, channels, height, width = x_t.shape
    device = x_t.device
    lengths = [int(v) for v in cond["lengths"].detach().cpu().tolist()]
    if len(lengths) != batch or any(v <= 0 or v > cond["embeds"].shape[1] for v in lengths):
        raise ValueError("FLUX.3 文本条件长度与图片批次不符。")
    tokens = height * width
    ids = torch.zeros(height, width, 4, device=device, dtype=torch.float32)
    ids[:, :, 1] = torch.arange(height, device=device)[:, None] * 0.5
    ids[:, :, 2] = torch.arange(width, device=device)[None, :] * 0.5
    text_ids, bounds, offset = [], [], 0
    for length in lengths:
        item = torch.zeros(length, 4, device=device, dtype=torch.float32)
        item[:, 3] = torch.arange(length, device=device)
        text_ids.append(item)
        bounds.append((offset, offset + length))
        offset += length
    data = {
        "image": x_t.flatten(2).transpose(1, 2).reshape(batch * tokens, channels).to(dtype),
        "image_cond": torch.zeros(0, channels, device=device, dtype=dtype),
        "text": torch.cat([cond["embeds"][i, :length] for i, length in enumerate(lengths)]).to(device, dtype),
    }
    return Packed(
        data=data,
        ids={"image": ids.reshape(tokens, 4).repeat(batch, 1),
             "image_cond": ids.new_zeros(0, 4), "text": torch.cat(text_ids)},
        timesteps={"image": t.to(device=device, dtype=dtype).repeat_interleave(tokens),
                   "image_cond": torch.zeros(0, device=device, dtype=dtype),
                   "text": torch.zeros(offset, device=device, dtype=dtype)},
        mode_indices={"image": [(i * tokens, (i + 1) * tokens) for i in range(batch)],
                      "image_cond": [(0, 0)] * batch, "text": bounds},
    )


class Flux3Family(ModelFamily):
    spec = ModelSpec(
        name="flux3", label="FLUX.3 Klein 5.5B", architecture="flux3-image-klein-base",
        latent=LatentSpec(128, 16, 1, "flux3-base-vae-mean-fp32-patch2-bn-v1"),
        text=TextSpec(4096, "flux3-qwen3vl-8layers-pad80-bf16-v1", pad_floor=False,
                      encoder_params=4_400_000_000),
        sampling=SamplingDefaults(steps=50, cfg=3.5, shift=6.93, sampler="euler"),
        capabilities=frozenset({"activation_checkpointing", "masked_loss", "online_text"}),
        attention_backends=("auto", "sdpa"),
        adapter_prefix="lora_unet",
        activation_units=26.0,
        weights=(
            ("dit_path", "Klein Base DiT", "FLUX.3 Image Klein Base 完整本地目录或原始 model.safetensors"),
            ("text_encoder_path", "文本编码器", "完整模型目录可留空；也可选择 Qwen3-VL-4B-Instruct 本地目录或配有 processor 的 BF16 单文件"),
            ("vae_path", "FLUX.3 VAE", "完整模型目录可留空；否则选择配套 ae.safetensors"),
            ("tokenizer_path", "文本处理器", "文本编码器目录内含时可留空；自定义时选择匹配的完整 processor 目录"),
        ),
        optional_weights=("text_encoder_path", "vae_path", "tokenizer_path"),
        directory_only_weights=("tokenizer_path",),
    )

    _paths = staticmethod(loading.component_paths)

    def validate_config(self, cfg):
        if not cfg.dit_path:
            return ["FLUX.3 请选择 Klein Base 主模型。"]
        if cfg.dtype == "fp16":
            return ["FLUX.3 主模型计算精度请选择 BF16 或 FP32。"]
        if cfg.attention not in self.spec.attention_backends:
            return ["FLUX.3 注意力请选择自动或 SDPA。"]
        if error := loading.runtime_error():
            return [error]
        try:
            _, dit, text, vae, processor = self._paths(cfg)
            loading.transformer_config(dit)
            Flux3Text.validate_paths(text, processor)
            Flux3Latent.validate_path(vae)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError) as error:
            return [str(error)]
        return []

    def training_options_errors(self, cfg):
        errors = super().training_options_errors(cfg)
        for failed, field, message in (
            (cfg.training.mode != "adapter", "training.mode", "FLUX.3 当前支持适配器训练。"),
            (cfg.training.train_text_encoder, "training.train_text_encoder", "FLUX.3 文本编码器保持冻结。"),
            (cfg.memory.compile, "memory.compile", "FLUX.3 训练暂不支持编译。"),
            (bool(cfg.memory.blocks_to_swap), "memory.blocks_to_swap", "FLUX.3 训练暂不支持块换出。"),
            (cfg.memory.activation_checkpointing not in {"none", "block"}, "memory.activation_checkpointing",
             "FLUX.3 梯度检查点请选择关闭或开启。"),
            (cfg.loop.distributed_strategy == "fsdp", "loop.distributed_strategy", "FLUX.3 训练暂不支持显存分片。"),
            (cfg.loop.gpu_count != 1, "loop.gpu_count", "FLUX.3 当前支持单卡训练。"),
            (cfg.adapter.algo == "tlora", "adapter.algo", "FLUX.3 变长序列尚不支持按图片时间步调整秩的 T-LoRA。"),
            (cfg.model.dtype == "fp16", "model.dtype", "FLUX.3 主模型计算精度请选择 BF16 或 FP32。"),
            (cfg.memory.base_precision == "fp16", "memory.base_precision", "FLUX.3 底模精度请选择保留原精度、BF16 或 FP32。"),
            (cfg.loop.mixed_precision == "fp16", "loop.mixed_precision", "FLUX.3 混合精度请选择 BF16 或关闭。"),
        ):
            if failed:
                errors.append({"loc": field, "msg": message})
        for index, rule in enumerate(cfg.adapter.rules):
            if rule.algo == "tlora":
                errors.append({"loc": f"adapter.rules.{index}.algo",
                               "msg": "FLUX.3 变长序列尚不支持按图片时间步调整秩的 T-LoRA。"})
        return errors

    def load(self, cfg, memory, *, device, dtype, backbone_device=None):
        problems = self.validate_config(cfg)
        if problems:
            raise ValueError("; ".join(problems))
        if (memory.base_precision not in {"auto", "bf16", "fp32"} or memory.compile or memory.blocks_to_swap
                or memory.activation_checkpointing not in {"none", "block"}):
            raise ValueError("FLUX.3 请选择未量化权重、关闭编译和块换出，并使用普通梯度检查点。")
        if dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("FLUX.3 主模型计算精度请选择 BF16 或 FP32。")
        _, dit, text_path, vae, processor = self._paths(cfg)
        config = loading.transformer_config(dit)
        text = Flux3Text(text_path, processor, device=device)
        with torch.device("meta"):
            backbone = DiT(config=config, checkpointing=memory.activation_checkpointing)
        return LoadedModel(
            backbone, text, Flux3Latent(vae, device=device),
            torch.device(device), dtype,
            extra={"dit_config": loading.config_dict(config), "dit_path": dit, "config": config,
                   "backbone_device": backbone_device or device, "checkpointing": memory.activation_checkpointing,
                   "text_encoder_weight_elements": text.weight_elements,
                   "materialized": False},
        )

    def materialize_backbone(self, loaded):
        if loaded.extra.get("materialized"):
            return
        loaded.text.unload()
        loaded.latent.unload()
        loaded.backbone = loading.load_transformer(
            loaded.extra["dit_path"], loaded.extra["config"], device=loaded.extra["backbone_device"],
            dtype=loaded.dtype, checkpointing=loaded.extra["checkpointing"],
        )
        loaded.extra["materialized"] = True

    def prepare_attention(self, loaded, configured, *, device, dtype, training, pinned=None):
        if configured not in {"auto", "sdpa"}:
            raise ValueError("FLUX.3 注意力请选择自动或 SDPA。")
        from ypuddin.models.attention_check import check_attention, grad_mode

        from .dit import Block, rope_embed

        config = loaded.extra["config"]
        hidden = sum(config.axes_dim) * 2

        def run(backend):
            with torch.autocast(torch.device(device).type, enabled=False), grad_mode(training):
                block = Block(hidden, 2, config.mlp_ratio).to(device=device, dtype=dtype)
                x = torch.randn(24, hidden, device=device, dtype=dtype, requires_grad=training)
                ids = torch.zeros(1, 24, 4, device=device)
                freqs = rope_embed(ids, config.axes_dim, config.theta)
                mods = tuple(torch.zeros_like(x) for _ in range(3))
                mods[-1].fill_(1)
                prediction = block(x, freqs, [(0, 16), (16, 24)], mods)
                if training:
                    prediction.float().square().mean().backward()

        loaded.extra["attention"] = check_attention(
            selected="sdpa", run=run, device=device, dtype=dtype, pinned=pinned,
        )

    def build_objective(self, loaded, cfg):
        from ypuddin.models.flux2.latent import patchify, unpatchify
        from ypuddin.objectives.flow import PackedFlowObjective

        return PackedFlowObjective(cfg, pack=patchify, unpack=unpatchify)

    def forward(self, loaded, x_t, t, cond, **extra):
        if not loaded.extra.get("materialized"):
            raise RuntimeError("FLUX.3 主模型尚未加载。")
        batch, channels, height, width = x_t.shape
        if channels != self.spec.latent.channels:
            raise ValueError("FLUX.3 需要 128 通道的归一化潜变量。")
        prediction = loaded.backbone(pack_inputs(x_t, t, cond, loaded.dtype))
        return prediction.reshape(batch, height * width, channels).transpose(1, 2).reshape_as(x_t)

    def sampling_prediction_pair(self, loaded, forward, cond, uncond, *, cfg, noise,
                                 reserve_memory=0, check=None):
        if cfg == 1.0:
            return None
        combined = combine_conditions([uncond, cond])

        def paired(x, t):
            if check:
                check()
            negative, positive = forward(torch.cat([x, x]), torch.cat([t, t]), combined).chunk(2)
            return positive, negative

        return paired

    def sample_latents(self, loaded, predict, shape, **options):
        # Both noise sources retain paired CFG, including shared samplers that call the branches separately.
        pair = options.get("predict_pair")
        if pair is not None:
            state = {}

            def positive(x, t):
                values = pair(x, t)
                state.update(x=x, t=t, values=values)
                return values[0]

            def negative(x, t):
                if state.get("x") is x and state.get("t") is t:
                    result = state["values"][1]
                    state.clear()
                    return result
                return pair(x, t)[1]

            predict, options["predict_uncond"] = positive, negative
        options["dtype"] = torch.float32
        return super().sample_latents(loaded, predict, shape, **options)

    def presets(self):
        blocks = ("content_mode_blocks.image.*", "txt_mode_blocks.*", "single_blocks.*")
        attn = tuple(f"{prefix}.{{q_proj,k_proj,v_proj,attn_out}}" for prefix in blocks)
        mlp = tuple(f"{prefix}.{{mlp_in,mlp_out}}" for prefix in blocks)
        return {
            "attn-only": TargetPreset("attn-only", attn, description="训练文生图分支的注意力投影。"),
            "attn-mlp": TargetPreset("attn-mlp", attn + mlp, description="训练文生图分支的注意力和前馈层。"),
        }

    def memory_layout_meta(self, backbone):
        blocks = []
        for index in range(len(backbone.txt_mode_blocks)):
            blocks.extend((backbone.content_mode_blocks["image"][index],
                           backbone.content_mode_blocks["image_cond"][index], backbone.txt_mode_blocks[index]))
        return MemoryLayout(blocks=[*blocks, *backbone.single_blocks])

    def memory_layout(self, loaded):
        return self.memory_layout_meta(loaded.backbone)

    def training_tokens_for_plan(self, image_tokens):
        return image_tokens + self.spec.text.max_len

    def meta_backbone(self, cfg):
        config = loading.transformer_config(self._paths(cfg)[1]) if cfg.dit_path else loading.BASE_CONFIG
        return DiT(config=config)

    def linear_module_names(self):
        with torch.device("meta"):
            model = self.meta_backbone(ModelConfig(family="flux3"))
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    def latent_fingerprint(self, cfg, *, dtype):
        path = self._paths(cfg)[3]
        return Flux3Latent(path).fingerprint if path.exists() else self.spec.latent.fingerprint


register("flux3", Flux3Family)
