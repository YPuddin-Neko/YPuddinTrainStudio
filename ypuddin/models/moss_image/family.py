"""MOSS-Image Base adapter training and its two-branch Euler sampler."""
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
from ypuddin.sampling.noise import SeedNoise

from . import loading
from .dit import MOSSImageTransformer, _Block, _RoPE
from .latent import MOSSImageLatent
from .objective import MOSSImageObjective
from .text import MOSSImageText


def sampling_sigmas(steps, shift=5.0, *, device="cpu"):
    if steps < 1 or shift <= 0:
        raise ValueError("MOSS-Image 采样步数和偏移必须大于 0。")
    values = torch.linspace(1, shift / (1000 + shift - 1), steps)
    sigmas = shift * values / (1 + (shift - 1) * values)
    training = torch.linspace(1, 0.001, 1000).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=999)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1))).to(device)


class MOSSImageFamily(ModelFamily):
    spec = ModelSpec(
        name="moss_image", label="MOSS-Image 5B", architecture="moss-image-base",
        latent=LatentSpec(128, 16, 1, "moss-image-small-vae-mode-fp32-patch2-bn-v1"),
        text=TextSpec(4096, "moss-image-qwen35-last-thinking-unpadded-4096-bf16-v1", pad_floor=False),
        sampling=SamplingDefaults(steps=50, cfg=3.0, shift=5.0, sampler="euler"),
        capabilities=frozenset({"activation_checkpointing", "masked_loss", "online_text"}),
        attention_backends=("auto", "sdpa", "flash_attn"), adapter_prefix="lora_unet",
        sampling_samplers=("euler",), sampling_schedulers=("uniform",),
        objective_timestep_sampling=("moss_image", "uniform", "logit_normal", "shift", "resolution_shift", "mode", "cosmap"),
        activation_units=42.0,
        weights=(("dit_path", "MOSS-Image Base DiT", "完整模型目录或 transformer 目录"),
                 ("text_encoder_path", "Qwen3.5 文本编码器", "完整模型内含时可留空；也可选择配套本地 text_encoder 目录"),
                 ("vae_path", "MOSS-Image VAE", "配套小解码器或标准 FLUX.2 VAE；完整模型内含时可留空"),
                 ("tokenizer_path", "分词器", "配套 tokenizer 目录；自动发现时可留空")),
        optional_weights=("text_encoder_path", "vae_path", "tokenizer_path"),
        directory_only_weights=("tokenizer_path",),
    )
    _paths = staticmethod(loading.component_paths)

    def validate_config(self, cfg):
        if not cfg.dit_path:
            return ["请选择 MOSS-Image Base 主模型。"]
        if cfg.dtype == "fp16":
            return ["MOSS-Image 主模型计算精度请选择 BF16 或 FP32。"]
        if cfg.attention not in self.spec.attention_backends:
            return ["MOSS-Image 注意力请选择自动、SDPA 或 FlashAttention 2。"]
        if error := loading.runtime_error():
            return [error]
        try:
            _, dit, text, vae, tokenizer = self._paths(cfg)
            loading.transformer_config(dit)
            MOSSImageText.validate_paths(text, tokenizer)
            MOSSImageLatent.validate_path(vae)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError) as error:
            return [str(error)]
        return []

    def training_options_errors(self, cfg):
        errors = super().training_options_errors(cfg)
        for failed, field, message in (
            (cfg.training.mode != "adapter", "training.mode", "MOSS-Image 当前支持适配器训练。"),
            (cfg.training.train_text_encoder, "training.train_text_encoder", "MOSS-Image 文本编码器保持冻结。"),
            (cfg.memory.compile, "memory.compile", "MOSS-Image 训练暂不支持编译。"),
            (bool(cfg.memory.blocks_to_swap), "memory.blocks_to_swap", "MOSS-Image 训练暂不支持块换出。"),
            (cfg.memory.activation_checkpointing not in {"none", "block"}, "memory.activation_checkpointing", "MOSS-Image 梯度检查点请选择关闭或开启。"),
            (cfg.loop.distributed_strategy == "fsdp", "loop.distributed_strategy", "MOSS-Image 训练暂不支持显存分片。"),
            (cfg.loop.gpu_count != 1, "loop.gpu_count", "MOSS-Image 当前支持单卡训练。"),
            (cfg.adapter.algo == "tlora", "adapter.algo", "MOSS-Image 当前不支持 T-LoRA。"),
            (cfg.model.dtype == "fp16", "model.dtype", "MOSS-Image 主模型计算精度请选择 BF16 或 FP32。"),
            (cfg.memory.base_precision not in {"auto", "bf16", "fp32"}, "memory.base_precision", "MOSS-Image 底模精度请选择保留原精度、BF16 或 FP32。"),
            (cfg.loop.mixed_precision == "fp16", "loop.mixed_precision", "MOSS-Image 混合精度请选择 BF16 或关闭。"),
        ):
            if failed:
                errors.append({"loc": field, "msg": message})
        for index, rule in enumerate(cfg.adapter.rules):
            if rule.algo == "tlora":
                errors.append({"loc": f"adapter.rules.{index}.algo", "msg": "MOSS-Image 当前不支持 T-LoRA。"})
        return errors

    def load(self, cfg, memory, *, device, dtype, backbone_device=None):
        if problems := self.validate_config(cfg):
            raise ValueError("; ".join(problems))
        if (memory.base_precision not in {"auto", "bf16", "fp32"} or memory.compile or memory.blocks_to_swap
                or memory.activation_checkpointing not in {"none", "block"}
                or dtype not in {torch.bfloat16, torch.float32}):
            raise ValueError("MOSS-Image 需要 BF16 或 FP32，关闭编译和块换出，使用普通梯度检查点。")
        _, dit, text, vae, tokenizer = self._paths(cfg)
        config = loading.transformer_config(dit)
        encoder = MOSSImageText(text, tokenizer, device=device)
        with torch.device("meta"):
            backbone = MOSSImageTransformer(**config, checkpointing=memory.activation_checkpointing)
        return LoadedModel(backbone, encoder, MOSSImageLatent(vae, device=device), torch.device(device), dtype,
                           extra={"dit_config": config, "config": config, "dit_path": dit,
                                  "backbone_device": backbone_device or device,
                                  "checkpointing": memory.activation_checkpointing,
                                  "text_encoder_weight_elements": encoder.weight_elements, "materialized": False})

    def materialize_backbone(self, loaded):
        if loaded.extra.get("materialized"):
            return
        loaded.text.unload()
        loaded.latent.unload()
        loaded.backbone = loading.load_transformer(loaded.extra["dit_path"], loaded.extra["config"],
                                                   device=loaded.extra["backbone_device"], dtype=loaded.dtype,
                                                   checkpointing=loaded.extra["checkpointing"])
        loaded.extra["materialized"] = True
        if loaded.extra.get("attention") == "flash_attn":
            loaded.backbone.set_attention_mode("flash_attn")

    def prepare_attention(self, loaded, configured, *, device, dtype, training, pinned=None):
        from ypuddin.models.attention_check import check_attention, grad_mode

        if configured not in self.spec.attention_backends:
            raise ValueError("MOSS-Image 注意力请选择自动、SDPA 或 FlashAttention 2。")
        config = loaded.extra["config"]
        head = config["dim"] // config["num_heads"]

        def run(backend):
            with torch.autocast(torch.device(device).type, enabled=False), grad_mode(training):
                dim = head * 2
                block = _Block(dim, int(dim * config["ffn_dim"] / config["dim"]), 2).to(device=device, dtype=dtype)
                modulation = torch.zeros(2, 6, dim, device=device, dtype=dtype).fill_(0.1)
                # Equal text lengths reach the selected backend; padded batches keep SDPA with their mask.
                for mode, lengths, mask in (
                    ("torch" if backend == "sdpa" else backend, [5, 5], None),
                    ("torch", [5, 3], (torch.arange(9, device=device)[None] < torch.tensor([9, 7], device=device)[:, None])[:, None, None]),
                ):
                    block.attn_mode = mode
                    x = torch.randn(2, 9, dim, device=device, dtype=dtype, requires_grad=training)
                    rope = _RoPE(head, config["rope_base"]).to(device)(2, 2, lengths, device).to(dtype)
                    y = block(x, rope, modulation, mask)
                    if training:
                        y.float().square().mean().backward()
        actual = check_attention(selected="flash_attn" if configured == "flash_attn" else "sdpa",
                                 run=run, device=device, dtype=dtype, pinned=pinned)
        loaded.extra["attention"] = actual
        if loaded.extra.get("materialized"):
            loaded.backbone.set_attention_mode("torch" if actual == "sdpa" else actual)

    def build_objective(self, loaded, cfg):
        return MOSSImageObjective(cfg)

    def forward(self, loaded, x_t, t, cond, **extra):
        if not loaded.extra.get("materialized"):
            raise RuntimeError("MOSS-Image 主模型尚未加载。")
        negative = "negative" in cond and bool(cond["negative"].all())
        skip = [22] if negative and len(loaded.backbone.layers) > 22 else []
        return loaded.backbone(x_t, t.float() * 1000, cond["embeds"], cond["mask"],
                               return_dict=False, skip_layers=skip)[0]

    def sampling_prediction_pair(self, loaded, forward, cond, uncond, *, cfg, noise, reserve_memory=0, check=None):
        if cfg == 1:
            return None
        negative = TextCond({**uncond.tensors, "negative": torch.ones(uncond.batch_size, dtype=torch.bool, device=uncond["embeds"].device)})

        def paired(x, t):
            if check:
                check()
            positive = forward(x, t, cond).float()
            if check:
                check()
            return positive, forward(x, t, negative).float()
        return paired

    def sample_latents(self, loaded, predict, shape, *, steps, shift=5.0, cfg=3.0, sampler="euler", scheduler="uniform",
                       predict_uncond=None, predict_pair=None, generator=None, device="cpu", dtype=torch.float32,
                       on_step=None, noise="comfyui", **options):
        if sampler != "euler" or scheduler != "uniform":
            raise ValueError("MOSS-Image 当前使用 Euler 与官方离散时间步日程。")
        x = SeedNoise(generator, noise, device).first(shape)
        sigmas = sampling_sigmas(steps, shift, device=device)
        for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            t = sigma.expand(shape[0])
            if cfg != 1:
                if predict_pair is None:
                    raise ValueError("MOSS-Image CFG 需要包含负分支跳层的配对预测。")
                positive, negative = predict_pair(x, t)
                prediction = negative.float() + cfg * (positive.float() - negative.float())
            else:
                prediction = predict(x, t).float()
            x = x + (following - sigma) * prediction
            if on_step:
                on_step(index + 1, len(sigmas) - 1)
        return x

    def presets(self):
        attn = ("layers.*.proj.{q,k,v,o}", "text_embedder.blocks.*.{qkv,proj}")
        mlp = ("layers.*.proj.{ffn_up_gate,ffn_down}", "text_embedder.blocks.*.{ffn_gate,ffn_up,ffn_down}")
        return {"attn-only": TargetPreset("attn-only", attn, description="训练注意力投影。"),
                "attn-mlp": TargetPreset("attn-mlp", attn + mlp, description="训练注意力和前馈层。")}

    def memory_layout_meta(self, backbone):
        return MemoryLayout(blocks=[*backbone.text_embedder.blocks, *backbone.layers])

    def memory_layout(self, loaded):
        return self.memory_layout_meta(loaded.backbone)

    def training_tokens_for_plan(self, image_tokens):
        return image_tokens + self.spec.text.max_len

    def meta_backbone(self, cfg):
        config = loading.transformer_config(self._paths(cfg)[1]) if cfg.dit_path else loading.BASE_CONFIG
        return MOSSImageTransformer(**config)

    def linear_module_names(self):
        with torch.device("meta"):
            model = self.meta_backbone(ModelConfig(family="moss_image"))
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    def latent_fingerprint(self, cfg, *, dtype):
        path = self._paths(cfg)[3]
        return MOSSImageLatent(path).fingerprint if path.exists() else self.spec.latent.fingerprint


register("moss_image", MOSSImageFamily)
