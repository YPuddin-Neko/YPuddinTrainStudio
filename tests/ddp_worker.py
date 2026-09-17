"""Subprocess-only acceptance hooks; never imported by production training."""

import json
import os
import sys

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from ypuddin.cli import main
from ypuddin.train.distributed import DistributedTrainer

original_run = DistributedTrainer.run
original_prepare = DistributedTrainer.prepare


def snapshot_prepare(self):
    original_prepare(self)
    if self.cfg.training.mode == "adapter" and not self.cfg.training.train_text_encoder:
        modules = {"backbone": self.loaded.backbone, **self.loaded.text.trainable_modules()}
        self._test_frozen = [
            (p, p.detach().clone())
            for module in modules.values()
            for p in module.parameters()
            if not p.requires_grad
        ]
        self._test_text_frozen = all(
            not p.requires_grad
            for name, module in modules.items()
            if name != "backbone"
            for p in module.parameters()
        )
        self._test_adapter_before = {k: t.clone() for k, t in self.adapters.training_state_dict().items()}


def snapshot_run(self):
    if self.is_primary and os.environ.get("YPUDDIN_TEST_PAUSE_STEP"):
        pause_step = int(os.environ["YPUDDIN_TEST_PAUSE_STEP"])

        def pause(event):
            if event["type"] == "step" and event["step"] == pause_step:
                control = self.run_dir / "control"
                control.mkdir(exist_ok=True)
                (control / "pause").touch()

        self.emitter.add_listener(pause)
    result = original_run(self)
    save_file(self.adapters.training_state_dict(), self.run_dir / f"rank-{self.distributed.rank}.safetensors")
    (self.run_dir / f"rank-{self.distributed.rank}.json").write_text(
        json.dumps(
            {
                "rank": self.distributed.rank,
                "pid": os.getpid(),
                "progress": self.progress.to_dict(),
                "samples": [index for batch in self.sampler.plan(0) for index in batch],
                "frozen_unchanged": all(
                    torch.equal(p, before) for p, before in getattr(self, "_test_frozen", [])
                ),
                "text_frozen": getattr(self, "_test_text_frozen", None),
                "adapter_updated": any(
                    not torch.equal(t, self._test_adapter_before[k])
                    for k, t in self.adapters.training_state_dict().items()
                )
                if hasattr(self, "_test_adapter_before")
                else None,
            }
        )
    )
    dist.barrier()
    return result


DistributedTrainer.run = snapshot_run
DistributedTrainer.prepare = snapshot_prepare
raise SystemExit(main(["train", sys.argv[1], "--device", "cpu"]))
