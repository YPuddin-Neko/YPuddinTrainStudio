"""Key-format conversion between kohya (``lora_unet_a_b_c.lora_down.weight``), ComfyUI/PEFT
(``diffusion_model.a.b.c.lora_A.weight``) and LyCORIS (``lycoris_a_b_c....``) layouts.

Underscores are ambiguous ("self_attn" vs "self.attn"), so the kohya -> dotted direction needs
the list of real module names of the target model.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

_KOHYA_TO_PEFT = {"lora_down.weight": "lora_A.weight", "lora_up.weight": "lora_B.weight"}
_PEFT_TO_KOHYA = {v: k for k, v in _KOHYA_TO_PEFT.items()}


def underscored_to_dotted(underscored: str, module_names: Iterable[str]) -> str | None:
    table = {n.replace(".", "_"): n for n in module_names}
    return table.get(underscored)


def kohya_to_comfy(
    tensors: dict[str, Tensor],
    module_names: Iterable[str],
    *,
    prefix: str = "lora_unet",
    comfy_prefix: str = "diffusion_model",
) -> dict[str, Tensor]:
    """LoRA keys become PEFT-style ``diffusion_model.<dotted>.lora_A/B.weight``; other algorithms
    keep kohya keys (ComfyUI resolves ``lora_unet_*`` LoKr/LoHa keys natively)."""
    names = list(module_names)
    out: dict[str, Tensor] = {}
    for key, t in tensors.items():
        module, _, suffix = key.partition(".")
        if not module.startswith(prefix + "_") or suffix not in _KOHYA_TO_PEFT and suffix != "alpha":
            out[key] = t
            continue
        dotted = underscored_to_dotted(module[len(prefix) + 1 :], names)
        if dotted is None:
            out[key] = t
            continue
        if suffix == "alpha":
            out[f"{comfy_prefix}.{dotted}.alpha"] = t
        else:
            out[f"{comfy_prefix}.{dotted}.{_KOHYA_TO_PEFT[suffix]}"] = t
    return out


def comfy_to_kohya(
    tensors: dict[str, Tensor],
    *,
    prefix: str = "lora_unet",
    comfy_prefixes: tuple[str, ...] = ("diffusion_model.", "transformer.", "net."),
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for key, t in tensors.items():
        stripped = key
        for cp in comfy_prefixes:
            if key.startswith(cp):
                stripped = key[len(cp) :]
                break
        else:
            out[key] = t
            continue
        for suffix in (*_PEFT_TO_KOHYA, "alpha", "lora_down.weight", "lora_up.weight"):
            if stripped.endswith("." + suffix):
                dotted = stripped[: -(len(suffix) + 1)]
                out[f"{prefix}_{dotted.replace('.', '_')}.{_PEFT_TO_KOHYA.get(suffix, suffix)}"] = t
                break
        else:
            out[key] = t
    return out


def lycoris_to_kohya(
    tensors: dict[str, Tensor], *, lycoris_prefix: str = "lycoris", prefix: str = "lora_unet"
) -> dict[str, Tensor]:
    out = {}
    for key, t in tensors.items():
        if key.startswith(lycoris_prefix + "_"):
            out[prefix + key[len(lycoris_prefix) :]] = t
        else:
            out[key] = t
    return out
