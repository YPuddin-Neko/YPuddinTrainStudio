"""Windows branch simulation plus a real, explicitly CPU-only two-rank protocol."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ypuddin.train import gloo_probe


@pytest.fixture
def mocked_cuda_group(monkeypatch):
    group = object()
    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_backend=lambda: "gloo",
        get_world_size=lambda group=None: 2,
        all_gather_object=lambda output, local, group=None: output.__setitem__(
            slice(None), [local.copy(), local.copy()]
        ),
        new_group=Mock(return_value=group),
        destroy_process_group=Mock(),
    )
    monkeypatch.setattr(gloo_probe, "dist", distributed)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    exercise = Mock(return_value={"checks": ["mocked"], "device": "cuda:1"})
    monkeypatch.setattr(gloo_probe, "_exercise_gloo", exercise)
    return distributed, group, exercise


@pytest.mark.parametrize("bf16", [True, False])
def test_cuda_probe_selects_real_requested_device_and_supported_dtypes(mocked_cuda_group, monkeypatch, bf16):
    distributed, group, exercise = mocked_cuda_group
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    result = gloo_probe.probe_windows_cuda_ddp("cuda:1", required_dtypes=(torch.float16,))
    expected = (torch.float32, torch.float16, torch.float64)
    exercise.assert_called_once_with(torch.device("cuda:1"), group, expected)
    assert result["checks"] == ["mocked"]
    assert distributed.new_group.call_args.kwargs["timeout"].total_seconds() == 30
    distributed.destroy_process_group.assert_called_once_with(group)


def test_required_unsupported_bf16_rejected_before_collectives(mocked_cuda_group, monkeypatch):
    distributed, _, exercise = mocked_cuda_group
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="BF16"):
        gloo_probe.probe_windows_cuda_ddp("cuda:0", required_dtypes=(torch.bfloat16,))
    distributed.new_group.assert_called_once()
    distributed.destroy_process_group.assert_called_once()
    exercise.assert_not_called()


def test_probe_failure_destroys_owned_group_and_preserves_operator_reason(mocked_cuda_group):
    distributed, group, exercise = mocked_cuda_group
    exercise.side_effect = RuntimeError("all_reduce/torch.float16: no CUDA backend")
    with pytest.raises(RuntimeError, match="all_reduce/torch.float16"):
        gloo_probe.probe_windows_cuda_ddp("cuda:0")
    distributed.destroy_process_group.assert_called_once_with(group)


def test_teardown_failure_cannot_hide_original_collective_error(mocked_cuda_group):
    distributed, _, exercise = mocked_cuda_group
    exercise.side_effect = RuntimeError("broadcast/torch.bfloat16: original failure")
    distributed.destroy_process_group.side_effect = RuntimeError("cleanup failure")
    with pytest.raises(RuntimeError, match="broadcast/torch.bfloat16: original failure"):
        gloo_probe.probe_windows_cuda_ddp("cuda:0")


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_probe_never_substitutes_cpu_for_cuda(mocked_cuda_group, device):
    distributed, _, exercise = mocked_cuda_group
    with pytest.raises(ValueError, match="rank-specific CUDA"):
        gloo_probe.probe_windows_cuda_ddp(device)
    distributed.new_group.assert_not_called()
    exercise.assert_not_called()


@pytest.mark.parametrize("change", ["uninitialized", "wrong_backend", "one_rank"])
def test_probe_requires_actual_multi_rank_gloo(mocked_cuda_group, change):
    distributed, _, _ = mocked_cuda_group
    if change == "uninitialized":
        distributed.is_initialized = lambda: False
    elif change == "wrong_backend":
        distributed.get_backend = lambda: "nccl"
    else:
        distributed.get_world_size = lambda group=None: 1
    with pytest.raises(ValueError, match="multi-rank Gloo"):
        gloo_probe.probe_windows_cuda_ddp("cuda:0")
    distributed.new_group.assert_not_called()


def test_collective_failure_names_the_failing_operation(monkeypatch):
    monkeypatch.setattr(gloo_probe.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(gloo_probe.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(gloo_probe.dist, "broadcast", Mock(side_effect=RuntimeError("backend unavailable")))
    with pytest.raises(RuntimeError, match="rank 0 / broadcast/torch.float32.*backend unavailable"):
        gloo_probe._exercise_gloo(torch.device("cpu"), object(), (torch.float32,))


def test_wrong_collective_value_fails_the_probe(monkeypatch):
    monkeypatch.setattr(gloo_probe.dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(gloo_probe.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(gloo_probe.dist, "broadcast", Mock())
    with pytest.raises(RuntimeError, match="broadcast result differs"):
        gloo_probe._exercise_gloo(torch.device("cpu"), object(), (torch.float32,))


@pytest.mark.parametrize("required", [(), (torch.bfloat16,)])
def test_mixed_bf16_capabilities_negotiate_uniform_sequence_or_rejection(mocked_cuda_group, required):
    distributed, _, exercise = mocked_cuda_group

    def mixed(output, local, group=None):
        output[:] = [{**local, "bf16": True}, {**local, "bf16": False}]

    distributed.all_gather_object = mixed
    if required:
        with pytest.raises(ValueError, match=r"rank \[1\].*BF16"):
            gloo_probe.probe_windows_cuda_ddp("cuda:0", required_dtypes=required)
        exercise.assert_not_called()
    else:
        gloo_probe.probe_windows_cuda_ddp("cuda:0")
        assert exercise.call_args.args[-1] == (torch.float32, torch.float16, torch.float64)
    distributed.destroy_process_group.assert_called_once()


def test_different_rank_dtype_requests_rejected_before_cuda_collectives(mocked_cuda_group):
    distributed, _, exercise = mocked_cuda_group

    def mismatch(output, local, group=None):
        output[:] = [local, {"bf16": True, "required": ["torch.bfloat16"]}]

    distributed.all_gather_object = mismatch
    with pytest.raises(ValueError, match="各 rank 请求的训练精度不同"):
        gloo_probe.probe_windows_cuda_ddp("cuda:0")
    exercise.assert_not_called()


@pytest.fixture
def simulated_windows_context(monkeypatch):
    from ypuddin.train import distributed as module

    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    for name, value in {
        "RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "12345",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("YPUDDIN_DISTRIBUTED_BACKEND", raising=False)
    monkeypatch.delenv("YPUDDIN_DDP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    distributed = SimpleNamespace(
        is_initialized=lambda: False,
        is_gloo_available=lambda: True,
        is_nccl_available=lambda: False,
        init_process_group=Mock(),
        rendezvous=Mock(side_effect=lambda *args, **kwargs: iter([(object(), 0, 2)])),
        destroy_process_group=Mock(),
    )
    monkeypatch.setattr(module, "dist", distributed)
    probe = Mock(return_value={"checks": ["mocked"], "device": "cuda:0"})
    monkeypatch.setattr(gloo_probe, "probe_windows_cuda_ddp", probe)
    return module, distributed, probe


def test_windows_context_probes_before_loading_or_writing_training_assets(
    simulated_windows_context, tmp_path, monkeypatch
):
    from ypuddin.config import TrainConfig

    module, distributed, probe = simulated_windows_context
    probe.side_effect = RuntimeError("all_reduce/torch.float16: unsupported")
    trainer = Mock(side_effect=AssertionError("must not construct training model"))
    monkeypatch.setattr(module, "DistributedTrainer", trainer)
    emitter = Mock(side_effect=AssertionError("must not write output"))
    monkeypatch.setattr(module, "Emitter", emitter)
    output = tmp_path / "unused-output"
    cfg = TrainConfig.model_validate(
        {"model": {"family": "toy"}, "loop": {"gpu_count": 2}, "checkpoint": {"output_dir": str(output)}}
    )
    with pytest.raises(RuntimeError, match="all_reduce/torch.float16"):
        module.distributed_train(cfg, device="cuda")
    assert distributed.init_process_group.call_args.args == ("gloo",)
    assert distributed.rendezvous.call_args.args == ("env://",)
    assert distributed.rendezvous.call_args.kwargs["timeout"].total_seconds() == 60
    assert distributed.init_process_group.call_args.kwargs["timeout"].total_seconds() == 1800
    assert distributed.init_process_group.call_args.kwargs["rank"] == 0
    assert distributed.init_process_group.call_args.kwargs["world_size"] == 2
    distributed.destroy_process_group.assert_called_once()
    trainer.assert_not_called()
    emitter.assert_not_called()
    assert not output.exists()


def test_windows_context_preserves_explicit_timeout_and_dtypes(simulated_windows_context, monkeypatch):
    module, distributed, probe = simulated_windows_context
    monkeypatch.setenv("YPUDDIN_DDP_TIMEOUT_SECONDS", "90")
    context = module.DistributedContext.initialize("cuda", required_dtypes=(torch.float16,))
    assert context.backend == "gloo" and context.device == torch.device("cuda:0")
    probe.assert_called_once_with(torch.device("cuda:0"), required_dtypes=(torch.float16,))
    assert distributed.init_process_group.call_args.kwargs["timeout"].total_seconds() == 90
    assert distributed.rendezvous.call_args.kwargs["timeout"].total_seconds() == 60


@pytest.mark.parametrize("existing", [None, ":16:8"])
def test_deterministic_workspace_set_before_first_cuda_use(simulated_windows_context, monkeypatch, existing):
    module, _, probe = simulated_windows_context
    if existing is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", existing)

    def available():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == (existing or ":4096:8")
        return True

    monkeypatch.setattr(torch.cuda, "is_available", available)
    module.DistributedContext.initialize("cuda", deterministic=True)
    probe.assert_called_once()


def test_rendezvous_failure_never_constructs_process_group_or_runs_probe(simulated_windows_context):
    module, distributed, probe = simulated_windows_context
    distributed.rendezvous.side_effect = RuntimeError("TCPStore connection timed out")
    with pytest.raises(RuntimeError, match="TCPStore connection timed out"):
        module.DistributedContext.initialize("cuda")
    distributed.init_process_group.assert_not_called()
    probe.assert_not_called()


def test_rendezvous_rank_mismatch_rejected_before_probe(simulated_windows_context):
    module, distributed, probe = simulated_windows_context
    distributed.rendezvous.side_effect = lambda *args, **kwargs: iter([(object(), 1, 2)])
    with pytest.raises(RuntimeError, match="rank/world differs"):
        module.DistributedContext.initialize("cuda")
    distributed.init_process_group.assert_not_called()
    probe.assert_not_called()


def test_windows_fsdp_rejected_before_communication_or_model_loading(simulated_windows_context):
    module, distributed, probe = simulated_windows_context
    with pytest.raises(ValueError, match="显存分片"):
        module.DistributedContext.initialize("cuda", strategy="fsdp")
    probe.assert_not_called()
    distributed.init_process_group.assert_not_called()


def test_real_two_rank_cpu_gloo_protocol_and_cleanup(tmp_path):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=127.0.0.1:0",
            "--rdzv_conf=is_host=true",
            "--local_addr=127.0.0.1",
            "--nproc_per_node=2",
            "-m",
            "Test.tests.gloo_probe_worker",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [json.loads((tmp_path / f"probe-rank-{rank}.json").read_text()) for rank in (0, 1)]
    for rank, report in enumerate(reports):
        assert report["rank"] == rank and report["world_size"] == 2
        assert report["device"] == "cpu" and report["backend"] == "gloo"
        assert report["rng_unchanged"] and report["default_group_destroyed"]
        assert report["mixed_capability_control_passed"]
        assert report["public_rendezvous_used"]
        assert not report["windows_validated"] and not report["cuda_validated"]
        assert report["checks"] == [
            "broadcast/torch.float32",
            "all_reduce/torch.float32",
            "all_reduce/int64/MAX",
            "all_reduce/int64/MIN",
            "all_gather/float32",
            "all_gather_object/control",
            "DDP constructor/forward/backward/update",
        ]
