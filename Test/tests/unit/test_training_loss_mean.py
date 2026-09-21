"""Running means count actual optimizer updates, including unlogged steps."""

from types import SimpleNamespace

import torch

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer


def test_loss_mean_counts_successful_steps_but_not_skipped_gradients(tmp_path):
    config = TrainConfig.model_validate(
        {"checkpoint": {"output_dir": str(tmp_path / "run")}, "loop": {"log_every": 2}}
    )
    trainer = Trainer(config, device="cpu")
    parameter = torch.nn.Parameter(torch.ones(1))
    trainer.adapters = SimpleNamespace(parameters=lambda: [parameter])
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.01)
    trainer.scheduler = None
    trainer.progress.total_steps = 10
    trainer._step_hooks = lambda _: None
    events = []
    trainer.emit = lambda kind, **payload: events.append({"type": kind, **payload})
    for loss, grad in [(1.0, 1.0), (9.0, 1.0), (1000.0, float("nan")), (2.0, 1.0), (4.0, 1.0)]:
        parameter.grad = torch.tensor([grad])
        trainer._optimizer_step(loss, 1.0)
    steps = [event for event in events if event["type"] == "step"]
    assert [event["step"] for event in steps] == [2, 4]
    assert [event["loss_mean"] for event in steps] == [5.0, 4.0]
    assert [event["loss_count"] for event in steps] == [2, 4]
    assert all(event["loss_mean_scope"] == "run" for event in steps)
    assert trainer.progress.extra["loss_sum"] == 16
    assert trainer.progress.extra["loss_count"] == 4
    assert steps[-1]["loss_mean"] != steps[-1]["loss_ema"]
