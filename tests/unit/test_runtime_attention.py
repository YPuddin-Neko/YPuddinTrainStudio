"""Real CPU attention math behind a fake accelerator verifies the bounded probe contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ypuddin.runtime_attention import AttentionEnvironmentError, attention_dependency_error, probe_sdpa

MISSING = "No matching libraries found for flash_attn_2_cuda*.so"


def fake_accelerator(*, hip="6.3.26093", bf16=True, failure=None):
    tensors, calls = [], []

    def randn(*shape, **kwargs):
        assert kwargs.pop("device") == "cuda:0"
        tensor = torch.randn(*shape, device="cpu", **kwargs)
        tensors.append(tensor)
        return tensor

    def sdpa(q, k, v, **kwargs):
        calls.append((list(q.shape), q.dtype))
        if failure and (error := failure(q)):
            raise error
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)

    fake = SimpleNamespace(
        __version__="2.7.1+das.opt1.dtk2604",
        version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda: "BW",
            is_bf16_supported=lambda: bf16,
            synchronize=Mock(),
        ),
        randn=randn,
        enable_grad=torch.enable_grad,
        isfinite=torch.isfinite,
        float16=torch.float16,
        bfloat16=torch.bfloat16,
        nn=SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=sdpa)),
    )
    return fake, tensors, calls


def test_probe_checks_half_precisions_and_distinct_head_dims_with_real_gradients():
    fake, tensors, calls = fake_accelerator()
    result = probe_sdpa(fake)
    assert result["status"] == "passed" and result["error"] is None
    assert result["device_name"] == "BW" and result["device"] == "cuda:0"
    assert calls == [
        ([1, 2, 32, 64], torch.float16),
        ([1, 2, 256, 128], torch.float16),
        ([1, 2, 32, 64], torch.bfloat16),
        ([1, 2, 256, 128], torch.bfloat16),
    ]
    assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in tensors)
    assert all(check["passed"] for check in result["checks"])
    assert max(t.numel() * t.element_size() for t in tensors) < 1024**2


def test_bfloat16_failure_is_visible_after_half_precision_success():
    fake, _, calls = fake_accelerator(
        failure=lambda q: RuntimeError(MISSING) if q.dtype == torch.bfloat16 else None
    )
    result = probe_sdpa(fake)
    assert result["status"] == "failed" and result["reason"] == "hip_sdpa_flash_library_missing"
    assert result["detail"] == MISSING and "厂商包" in result["error"]
    assert result["checks"][-1]["dtype"] == "bf16" and not result["checks"][-1]["passed"]
    assert len(calls) == 3  # Stop after the first failing kernel; do not retry another backend.


@pytest.mark.parametrize(
    "hip,detail",
    [
        (None, MISSING),
        ("6.3", "HIP out of memory"),
        ("6.3", "illegal memory access"),
        ("6.3", "undefined symbol in unrelated.so"),
    ],
)
def test_unrelated_runtime_errors_keep_their_original_meaning(hip, detail):
    fake, _, calls = fake_accelerator(hip=hip, failure=lambda q: RuntimeError(detail))
    result = probe_sdpa(fake)
    assert result["status"] == "failed" and result["reason"] == "sdpa_probe_failed"
    assert result["error"] == detail and len(calls) == 1
    assert attention_dependency_error(RuntimeError(detail), hip_runtime=hip) is None


def test_unsupported_bfloat16_is_not_falsely_tested_and_unavailable_gpu_never_allocates():
    fake, tensors, _ = fake_accelerator(bf16=False)
    result = probe_sdpa(fake)
    assert result["status"] == "passed" and len(result["checks"]) == 2
    assert all(check["dtype"] == "fp16" for check in result["checks"])
    fake.cuda.is_available = lambda: False
    tensors.clear()
    assert probe_sdpa(fake)["status"] == "not_tested"
    assert not tensors


def test_attention_shim_has_precise_dependency_diagnosis_without_catching_oom(monkeypatch):
    from ypuddin.models.anima.vendor.attention import _sdpa

    monkeypatch.setattr(torch.version, "hip", "6.3.26093")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.randn(1, 2, 3, 8)
    native = RuntimeError(MISSING)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", Mock(side_effect=native))
    with pytest.raises(AttentionEnvironmentError) as error:
        _sdpa(q, q, q, "sdpa")
    assert error.value.__cause__ is native and error.value.detail == MISSING
    assert "运行环境" in str(error.value)
    oom = RuntimeError("HIP out of memory")
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", Mock(side_effect=oom))
    with pytest.raises(RuntimeError) as error:
        _sdpa(q, q, q, "sdpa")
    assert error.value is oom


def test_cli_prints_known_dependency_guidance_but_keeps_verbose_and_unknown_errors(monkeypatch, capsys):
    import ypuddin.train
    from ypuddin import cli

    monkeypatch.setattr(cli, "_load", lambda args: object())
    monkeypatch.setenv("RANK", "0")
    error = AttentionEnvironmentError(MISSING)
    monkeypatch.setattr(ypuddin.train, "train", Mock(side_effect=error))
    args = SimpleNamespace(device="cuda", verbose=False)
    assert cli.cmd_train(args) == 1
    assert "厂商包" in capsys.readouterr().err
    args.verbose = True
    with pytest.raises(AttentionEnvironmentError):
        cli.cmd_train(args)
    args.verbose = False
    monkeypatch.setattr(ypuddin.train, "train", Mock(side_effect=RuntimeError("HIP out of memory")))
    with pytest.raises(RuntimeError, match="out of memory"):
        cli.cmd_train(args)
