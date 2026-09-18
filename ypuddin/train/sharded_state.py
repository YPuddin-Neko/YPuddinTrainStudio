"""FSDP2 checkpoints without materializing the whole model on an accelerator.

Every rank calls these functions. A single DTensor is gathered at a time and
immediately copied to CPU; only rank zero keeps the full state or writes files.
Restore slices CPU tensors before transferring local shards to the device. The
format deliberately refuses cross-world-size or cross-strategy exact resume.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from ypuddin.config.compute_policy import validate_resume_compute_policy
from ypuddin.config.io import config_hash, load_config

from .state import Progress
from .training_modes import FullTrainingSet, save_model_artifact

_FORMAT = 4


def _rank_world() -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError("分片训练状态需要已初始化的分布式进程组")
    return dist.get_rank(), dist.get_world_size()


def _primary_call(function):
    rank, _ = _rank_world()
    result = [None, None]
    if rank == 0:
        try:
            result[0] = function()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(f"分片训练状态文件操作失败：{result[1]}")
    return result[0]


def _collective_check(function):
    """A rank-local validation failure must reach peers before tensor collectives."""
    _, world = _rank_world()
    error = None
    result = None
    try:
        result = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError("分片训练状态检查失败：" + "; ".join(str(e) for e in errors if e))
    return result


def _placements(tensor: DTensor) -> list[dict[str, Any]]:
    if tensor.device_mesh.ndim != 1:
        raise ValueError("分片状态目前仅支持一维显卡网格")
    if tensor.device_mesh.size() != dist.get_world_size():
        raise ValueError("分片状态显卡网格必须覆盖整个训练进程组")
    placements = []
    for placement in tensor.placements:
        if isinstance(placement, Shard):
            placements.append({"kind": "shard", "dim": placement.dim})
        elif isinstance(placement, Replicate):
            placements.append({"kind": "replicate"})
        else:
            raise ValueError("保存前必须完成优化器状态的跨卡归约，不能保存 Partial 张量")
    return placements


def _tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "stride": list(tensor.stride())}
    if isinstance(tensor, DTensor):
        result["placements"] = _placements(tensor)
    else:
        result["device_type"] = tensor.device.type
    return result


def _description(value):
    if isinstance(value, torch.Tensor):
        return _tensor_layout(value)
    if isinstance(value, dict):
        return {str(k): _description(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_description(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else type(value).__name__


def _same_structure(value):
    description = _collective_check(lambda: _description(value))
    _, world = _rank_world()
    descriptions = [None] * world
    dist.all_gather_object(descriptions, description)
    if any(item != descriptions[0] for item in descriptions[1:]):
        raise ValueError("各显卡的模型或优化器状态结构不一致，不能保存分片状态")


def _model_state(modules: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
    state = {}
    for component, module in sorted(modules.items()):
        for name, tensor in module.state_dict().items():
            native = f"{component}.{FullTrainingSet._native_key(name)}"
            if native in state:
                raise ValueError(f"模型包含重复权重名称：{native}")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"分片模型状态必须是张量：{native}")
            state[native] = tensor
    return dict(sorted(state.items()))


def _parameter_names(modules: dict[str, nn.Module]) -> dict[int, str]:
    names = {}
    for component, module in sorted(modules.items()):
        for name, parameter in module.named_parameters():
            names.setdefault(id(parameter), f"{component}.{FullTrainingSet._native_key(name)}")
    return names


def _optimizer_state(modules, optimizer):
    names = _parameter_names(modules)
    groups = []
    for group in optimizer.param_groups:
        try:
            params = [names[id(parameter)] for parameter in group["params"]]
        except KeyError as exc:
            raise ValueError("优化器引用了未注册的模型参数") from exc
        groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": params})
    all_names = [name for group in groups for name in group["params"]]
    if len(all_names) != len(set(all_names)):
        raise ValueError("分片优化器包含重复的参数绑定")
    state = {names[id(parameter)]: value for parameter, value in optimizer.state.items()}
    return {"state": dict(sorted(state.items())), "param_groups": groups}


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor | None:
    rank, _ = _rank_world()
    # Keep no full accelerator tensors in a list/dict: full_tensor is scoped to
    # this call and the next collective cannot overlap its lifetime.
    full = tensor.detach().full_tensor() if isinstance(tensor, DTensor) else tensor.detach()
    if rank == 0:
        return full.to(device="cpu", copy=True).contiguous()
    return None


def gather_full_training_state(modules: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
    """Collect native full weights on CPU on rank zero; other ranks return {}."""
    state = _collective_check(lambda: _model_state(modules))
    _same_structure(state)
    result = {}
    for name, tensor in state.items():
        cpu = _cpu_tensor(tensor)
        if cpu is not None:
            result[name] = cpu
    return result


def _pack(value):
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "layout": _tensor_layout(value), "value": _cpu_tensor(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "value": [(k, _pack(v)) for k, v in value.items()]}
    if isinstance(value, list):
        return {"kind": "list", "value": [_pack(v) for v in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [_pack(v) for v in value]}
    return {"kind": "value", "value": copy.deepcopy(value)}


def _local_tensor(full, layout, *, mesh, device):
    placements = layout.get("placements")
    if placements is None:
        # Adam's uncapturable step counter must stay on CPU; moments follow
        # their recorded accelerator placement through optimizer.load_state_dict.
        target = "cpu" if layout.get("device_type") == "cpu" else device
        return full.to(device=target, copy=True)
    if mesh is None or mesh.ndim != 1:
        raise ValueError("恢复分片优化器状态需要一维设备网格")
    placement = placements[0]
    if len(placements) != 1:
        raise ValueError("分片状态目前仅支持一维显卡网格")
    if placement["kind"] == "shard":
        axis = placement["dim"]
        # torch.chunk follows DTensor's uneven sharding rule (tensor_split does
        # not: e.g. 5 rows / 4 ranks is 2,2,1,0 rather than 2,1,1,1).
        chunks = torch.chunk(full, mesh.size(), dim=axis)
        rank = mesh.get_local_rank()
        if rank < len(chunks):
            local = chunks[rank]
        else:
            shape = list(full.shape)
            shape[axis] = 0
            local = full.new_empty(shape)
        placement_object = Shard(axis)
    elif placement["kind"] == "replicate":
        local, placement_object = full, Replicate()
    else:
        raise ValueError("不支持的分片布局")
    local = local.contiguous().to(device=device, copy=True)
    return DTensor.from_local(
        local,
        mesh,
        [placement_object],
        run_check=False,
        shape=torch.Size(layout["shape"]),
        stride=tuple(layout["stride"]),
    )


def _unpack(node, *, mesh, device):
    kind, value = node["kind"], node["value"]
    if kind == "tensor":
        return _local_tensor(value, node["layout"], mesh=mesh, device=device)
    if kind == "dict":
        return {k: _unpack(v, mesh=mesh, device=device) for k, v in value}
    if kind in {"list", "tuple"}:
        values = [_unpack(v, mesh=mesh, device=device) for v in value]
        return tuple(values) if kind == "tuple" else values
    if kind != "value":
        raise ValueError("不支持的优化器状态内容")
    return copy.deepcopy(value)


def _atomic_directory(path: Path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + path.name + ".tmp-", dir=path.parent))
    backup = None
    try:
        write(temporary)
        # Only complete checkpoints are published. Preserve a previous complete
        # checkpoint if writing fails; never delete it before the new one exists.
        (temporary / "complete.json").write_text(json.dumps({"format": _FORMAT}), encoding="utf-8")
        if path.exists():
            backup = temporary.with_name(temporary.name + ".previous")
            path.rename(backup)
        try:
            temporary.rename(path)
        except Exception:
            if backup is not None:
                backup.rename(path)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return str(path)


def save_sharded_checkpoint(
    path: str | Path,
    *,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler_state: dict[str, Any],
    progress: Progress,
    rng: dict[str, Any],
    batch_size: int,
    grad_accum: int,
    training_kind: str = "full-model",
    adapter_contract: dict[str, Any] | None = None,
    adapter_metadata: dict[str, str] | None = None,
    config_hash: str = "",
    dataset_fingerprint: str = "",
    model_identity: str = "",
    scheduler_config: dict[str, Any] | None = None,
) -> Path:
    """Save full-model format 4; all ranks call, rank zero writes atomically."""
    _, world = _rank_world()
    optimizer_state = _collective_check(lambda: _optimizer_state(modules, optimizer))
    _same_structure(optimizer_state)
    tensors = gather_full_training_state(modules)
    packed_optimizer = _pack(optimizer_state)
    ranks = [None] * world
    dist.all_gather_object(ranks, {"rng": rng, "sampler": sampler_state, "progress": progress.to_dict()})
    if any(item["progress"] != ranks[0]["progress"] for item in ranks[1:]):
        raise ValueError("各显卡的训练进度不一致，不能保存分片状态")
    meta = {
        "format": _FORMAT,
        "training_kind": training_kind,
        "adapter_contract": adapter_contract,
        "strategy": "fsdp2",
        "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "model_layout": {name: _tensor_layout(value) for name, value in _model_state(modules).items()},
        "world_size": world,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "progress": progress.to_dict(),
        "sampler": ranks[0]["sampler"],
        "sampler_ranks": [item["sampler"] for item in ranks],
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "model_identity": model_identity,
    }
    if scheduler_config is not None:
        meta["scheduler_contract"] = {
            "config": copy.deepcopy(scheduler_config),
            "total_steps": progress.total_steps,
        }
    checkpoint_rng = {
        **ranks[0]["rng"],
        "distributed": {
            "world_size": world,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "ranks": [item["rng"] for item in ranks],
        },
    }

    def write(temporary):
        save_file(tensors, str(temporary / "model.safetensors"), metadata=adapter_metadata or {})
        torch.save(packed_optimizer, temporary / "optimizer.pt")
        torch.save(scheduler.state_dict() if scheduler is not None else {}, temporary / "scheduler.pt")
        torch.save(checkpoint_rng, temporary / "rng.pt")
        (temporary / "state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return Path(_primary_call(lambda: _atomic_directory(Path(path), write)))


def read_sharded_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read metadata only; no live parameter, optimizer or random state changes."""
    path = Path(path)
    meta = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if meta.get("format") != _FORMAT or meta.get("strategy") != "fsdp2":
        raise ValueError("完整训练状态的分布方式不同，请使用导出的完整模型权重开始新的训练任务")
    if meta.get("training_kind") not in {"full-model", "adapter"}:
        raise ValueError("分片训练状态必须包含完整模型")
    if not (path / "complete.json").is_file():
        raise ValueError("分片训练状态未完整写入，不能恢复")
    if json.loads((path / "complete.json").read_text(encoding="utf-8")).get("format") != _FORMAT:
        raise ValueError("分片训练状态完成标记无效")
    return meta


def _scheduler_contract(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    contract = meta.get("scheduler_contract")
    if contract is not None:
        return contract
    # r2 saved the LambdaLR state but not its Python closure. Only trust the
    # original run config if its fingerprint matches this checkpoint, never a
    # newly edited config or the current resume job's output/resume paths.
    try:
        original = load_config(path.parent / "config.toml")
        if not meta.get("config_hash") or config_hash(original) != meta["config_hash"]:
            raise ValueError("原训练配置已修改或无法确认来源")
        return {
            "config": original.scheduler.model_dump(mode="json"),
            "total_steps": meta["progress"]["total_steps"],
        }
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            "旧分片训练状态缺少学习率调度配置，无法验证精确恢复；"
            "请保留原训练目录中未修改的 config.toml，或使用完整模型权重开始新任务。"
        ) from exc


def read_sharded_scheduler_contract(path: str | Path) -> dict[str, Any]:
    """Capture the original recipe before a same-directory resume writes its config."""
    path = Path(path)
    return _scheduler_contract(path, read_sharded_checkpoint_metadata(path))


def load_sharded_checkpoint(
    path: str | Path,
    *,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    expected_training_kind: str = "full-model",
    expected_adapter_contract: dict[str, Any] | None = None,
    expected_world_size: int | None = None,
    expected_batch_size: int | None = None,
    expected_grad_accum: int | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_model_identity: str | None = None,
    expected_deterministic: bool | None = None,
    expected_compute_policy: dict[str, Any] | None = None,
    expected_compute_runtime: dict[str, Any] | None = None,
    expected_scheduler_config: dict[str, Any] | None = None,
    expected_total_steps: int | None = None,
    legacy_scheduler_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate before mutation, then restore native tensors and optimizer shards.

    The caller applies returned progress/scheduler/sampler/RNG using the normal
    Trainer contracts. This function intentionally does not consume or restore RNG.
    """
    rank, world = _rank_world()
    path = Path(path)

    def validate():
        meta = read_sharded_checkpoint_metadata(path)
        checks = {
            "training_kind": expected_training_kind,
            "adapter_contract": expected_adapter_contract,
            "world_size": world if expected_world_size is None else expected_world_size,
            "batch_size": expected_batch_size,
            "grad_accum": expected_grad_accum,
            "dataset_fingerprint": expected_dataset_fingerprint,
            "model_identity": expected_model_identity,
        }
        if checks["world_size"] != world:
            raise ValueError("当前进程组与预期显卡数量不一致")
        for key, expected in checks.items():
            if expected is not None and meta.get(key) != expected:
                raise ValueError(f"分片训练状态的 {key} 与当前训练设置不同，不能精确恢复")
        if expected_deterministic is not None:
            if meta["progress"].get("extra", {}).get("deterministic") != expected_deterministic:
                raise ValueError("分片训练状态的可复现计算设置与当前训练不同")
        validate_resume_compute_policy(
            expected_compute_policy, meta["progress"].get("extra", {}).get("compute_policy")
        )
        if expected_compute_policy is not None:
            from .reproducibility import validate_compute_runtime

            validate_compute_runtime(
                expected_compute_runtime, meta["progress"].get("extra", {}).get("compute_runtime")
            )
        if expected_scheduler_config is not None or expected_total_steps is not None:
            contract = meta.get("scheduler_contract")
            if contract is None:
                contract = legacy_scheduler_contract or _scheduler_contract(path, meta)
            if not isinstance(contract, dict) or not isinstance(contract.get("config"), dict):
                raise ValueError("分片训练状态的学习率调度配置无效，不能精确恢复")
            if expected_scheduler_config is not None and contract["config"] != expected_scheduler_config:
                raise ValueError("学习率调度配置与原训练不同，不能精确恢复；请恢复原设置或开始新任务")
            if (
                contract.get("total_steps") != meta["progress"].get("total_steps")
                or expected_total_steps is not None
                and contract.get("total_steps") != expected_total_steps
            ):
                raise ValueError("总训练步数与原训练不同，不能精确恢复学习率调度；请恢复原设置或开始新任务")
        if meta.get("optimizer_class") != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}":
            raise ValueError("分片训练状态的优化器类型与当前训练不同")
        states = _model_state(modules)
        with safe_open(path / "model.safetensors", framework="pt", device="cpu") as archive:
            if set(archive.keys()) != set(states):
                raise ValueError("分片训练状态的模型参数名称不同")
            for name, target in states.items():
                if meta.get("model_layout", {}).get(name) != _tensor_layout(target):
                    raise ValueError(f"分片训练状态的模型布局不同：{name}")
                saved = archive.get_slice(name)
                if list(saved.get_shape()) != list(target.shape):
                    raise ValueError(f"分片训练状态的模型参数形状不同：{name}")
                # Also validate dtype before the first parameter mutation.
                if archive.get_tensor(name).dtype != target.dtype:
                    raise ValueError(f"分片训练状态的模型参数精度不同：{name}")
        packed = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False, mmap=True)
        group_node = dict(packed["value"])["param_groups"]
        groups = _unpack(group_node, mesh=None, device="cpu")
        expected_groups = _optimizer_state(modules, optimizer)["param_groups"]
        if [g["params"] for g in groups] != [g["params"] for g in expected_groups]:
            raise ValueError("分片训练状态的优化器参数绑定或分组不同")
        rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
        saved_rng = rng.get("distributed", {})
        if (
            any(saved_rng.get(key) != meta[key] for key in ("world_size", "batch_size", "grad_accum"))
            or len(saved_rng.get("ranks", [])) != world
        ):
            raise ValueError("分片训练状态缺少各显卡的随机状态")
        if len(meta.get("sampler_ranks", [])) != world:
            raise ValueError("分片训练状态缺少各显卡的数据采样位置")
        scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False)
        return meta, states, packed, rng, scheduler

    meta, states, packed_optimizer, rng, scheduler = _collective_check(validate)
    with torch.no_grad(), safe_open(path / "model.safetensors", framework="pt", device="cpu") as archive:
        for name, target in states.items():
            full = archive.get_tensor(name)
            layout = _tensor_layout(target)
            restored = _local_tensor(
                full,
                layout,
                mesh=target.device_mesh if isinstance(target, DTensor) else None,
                device=target.device,
            )
            target.copy_(restored)
    mesh = next(
        (p.device_mesh for m in modules.values() for p in m.parameters() if isinstance(p, DTensor)), None
    )
    device = next(iter(optimizer.param_groups[0]["params"])).device
    restored_optimizer = _unpack(packed_optimizer, mesh=mesh, device=device)
    # Optimizer.load_state_dict binds each saved group to the already-created
    # parameters in order. Native names above are verified before this mapping.
    indices = {
        name: index
        for index, name in enumerate(
            name for group in restored_optimizer["param_groups"] for name in group["params"]
        )
    }
    restored_optimizer["state"] = {
        indices[name]: value for name, value in restored_optimizer["state"].items()
    }
    for group in restored_optimizer["param_groups"]:
        group["params"] = [indices[name] for name in group["params"]]
    optimizer.load_state_dict(restored_optimizer)
    return {
        **meta,
        "progress": Progress(**meta["progress"]),
        "sampler": meta["sampler_ranks"][rank],
        "rng": rng,
        "scheduler": scheduler,
    }


def export_sharded_model_artifact(path: Path, training: FullTrainingSet, cfg, loaded) -> Path:
    """All ranks gather; only rank zero invokes the existing native exporter."""
    tensors = gather_full_training_state(training.modules)
    return Path(_primary_call(lambda: str(save_model_artifact(path, training, cfg, loaded, tensors=tensors))))
