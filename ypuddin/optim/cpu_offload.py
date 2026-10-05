"""AdamW updates with parameter copies and optimizer state resident on the CPU."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


class CPUOffloadAdamW(Optimizer):
    """Expose live parameters to AMP/DDP while native AdamW updates CPU FP32 copies.

    Gradient unscaling, accumulation and clipping happen on the live parameters
    before ``step``. Transfers are synchronous: CPU updates finish before weights
    are copied back for the next forward or checkpoint.
    """

    def __init__(self, param_groups: list[dict[str, Any]], **kwargs):
        groups = [{**group, "params": list(group["params"])} for group in param_groups]
        parameters = [parameter for group in groups for parameter in group["params"]]
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("CPU optimizer offload requires unique parameters across groups")
        for group in groups:
            self._validate_options({**kwargs, **group})
        for parameter in parameters:
            if isinstance(parameter, DTensor):
                raise ValueError("CPU optimizer offload does not support FSDP2 parameters")
            if parameter.dtype != torch.float32 or parameter.device.type not in {"cpu", "cuda", "mps"}:
                raise ValueError("CPU optimizer offload requires ordinary FP32 training parameters")
        shadows = {parameter: parameter.detach().to(device="cpu", copy=True) for parameter in parameters}
        cpu_groups = [{**group, "params": [shadows[parameter] for parameter in group["params"]]} for group in groups]
        inner = torch.optim.AdamW(cpu_groups, **kwargs)
        super().__init__(groups, inner.defaults)
        self._cpu_optimizer = inner
        self._cpu_parameters = shadows
        self._refresh_state()

    @staticmethod
    def _validate_options(options: dict[str, Any]) -> None:
        for name in ("capturable", "differentiable", "fused"):
            if options.get(name):
                raise ValueError(f"CPU optimizer offload does not support AdamW {name}=True")
        if isinstance(options.get("lr"), Tensor):
            raise ValueError("CPU optimizer offload requires a scalar learning rate")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if hasattr(self, "_cpu_optimizer"):
            raise ValueError("CPU optimizer offload parameter groups cannot change after initialization")
        super().add_param_group(param_group)

    def _sync_groups(self) -> None:
        for group, cpu_group in zip(self.param_groups, self._cpu_optimizer.param_groups, strict=True):
            self._validate_options(group)
            if len(group["params"]) != len(cpu_group["params"]) or any(
                self._cpu_parameters.get(parameter) is not shadow
                for parameter, shadow in zip(group["params"], cpu_group["params"], strict=True)
            ):
                raise ValueError("CPU optimizer offload parameter groups changed after initialization")
            cpu_group.update({key: value for key, value in group.items() if key != "params"})

    def _refresh_state(self) -> None:
        self.state = defaultdict(dict, {
            parameter: self._cpu_optimizer.state[shadow]
            for parameter, shadow in self._cpu_parameters.items()
            if shadow in self._cpu_optimizer.state
        })

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync_groups()
        active = []
        try:
            for parameter, shadow in self._cpu_parameters.items():
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("CPU AdamW does not support sparse gradients")
                shadow.grad = parameter.grad.detach().to(device="cpu", copy=True)
                active.append((parameter, shadow))
            self._cpu_optimizer.step()
            for parameter, shadow in active:
                parameter.copy_(shadow)
        finally:
            self._cpu_optimizer.zero_grad(set_to_none=True)
            self._refresh_state()
        return loss

    def state_dict(self) -> dict[str, Any]:
        self._sync_groups()
        state = self._cpu_optimizer.state_dict()
        state["cpu_offload"] = {"version": 1, "optimizer": "adamw"}
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        marker = state_dict.get("cpu_offload")
        if marker is not None and marker != {"version": 1, "optimizer": "adamw"}:
            raise ValueError("Unsupported CPU optimizer offload checkpoint format")
        for group in state_dict["param_groups"]:
            self._validate_options(group)
        state = {key: value for key, value in state_dict.items() if key != "cpu_offload"}
        self._cpu_optimizer.load_state_dict(state)
        for group, cpu_group in zip(self.param_groups, self._cpu_optimizer.param_groups, strict=True):
            parameters = group["params"]
            group.clear()
            group.update({key: value for key, value in cpu_group.items() if key != "params"}, params=parameters)
        # The trainer restores model weights before optimizer state. Keeping this
        # copy on CPU also avoids the generic optimizer loader moving moments to GPU.
        for parameter, shadow in self._cpu_parameters.items():
            shadow.copy_(parameter.detach().to(device="cpu"))
        self._refresh_state()
