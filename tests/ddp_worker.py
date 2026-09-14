"""Subprocess-only acceptance hooks; never imported by production training."""

import json
import os
import sys

import torch.distributed as dist
from safetensors.torch import save_file

from ypuddin.cli import main
from ypuddin.train.distributed import DistributedTrainer

original_run = DistributedTrainer.run


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
            }
        )
    )
    dist.barrier()
    return result


DistributedTrainer.run = snapshot_run
raise SystemExit(main(["train", sys.argv[1], "--device", "cpu"]))
