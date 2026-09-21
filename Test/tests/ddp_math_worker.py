"""Exercise distributed image-weighted accumulation against an independent SGD oracle."""

import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ypuddin.config import TrainConfig
from ypuddin.data import BucketBatchSampler
from ypuddin.train.distributed import DistributedContext, DistributedTrainer, _TrainingGraph
from ypuddin.train.training_modes import FullTrainingSet

out = Path(sys.argv[1])
reject_first = "nonfinite" in sys.argv[2:]
native = "native" in sys.argv[2:]
context = DistributedContext.initialize("cpu")
try:
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(out)}],
                "batch_size": 2,
                "resolution_mode": "native" if native else "bucket",
            },
            "training": {"mode": "full"},
            "memory": {"base_precision": "fp32"},
            "loop": {"epochs": 1, "grad_accum": 2},
            "checkpoint": {"output_dir": str(out), "save_every_epochs": None, "save_on_finish": False},
            "optimizer": {"lr": 0.001, "grad_clip_norm": 0},
            "sampling": {"enabled": False},
            "validation": {"enabled": False},
        }
    )
    trainer = DistributedTrainer(cfg, context=context)
    linear = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    linear.weight.data.fill_(0.5)
    trainer.loaded = SimpleNamespace(backbone=linear)
    trainer.adapters = FullTrainingSet({"backbone": linear})
    # FullTrainingSet uses fp32 intentionally, and the oracle below uses the same precision.
    trainer.optimizer = torch.optim.SGD(linear.parameters(), lr=0.001)
    trainer.scheduler = None
    trainer.sampler = BucketBatchSampler([(1, 1)] * 3, 1)
    trainer.bundle = SimpleNamespace(train=SimpleNamespace(set_epoch=lambda _: None))
    inputs = ([[1.0, 2.0], [3.0], [4.0]], [[5.0], [6.0, 7.0], [8.0, 9.0]])
    trainer.loader = [
        {"caption": [""] * len(values), "x": torch.tensor(values).view(-1, 1), "micro": micro}
        for micro, values in enumerate(inputs[context.rank])
    ]
    if native:
        for batch in trainer.loader:
            # Deliberately unequal groups per rank and logical batch, including
            # a tail. Only real images enter the independent global SGD oracle.
            groups = batch["x"].split(1) if context.rank == 1 else [batch["x"]]
            batch["microbatches"] = [
                {"caption": [""] * len(x), "x": x, "micro": batch["micro"], "bucket": (1, 1)} for x in groups
            ]
    trainer.progress.total_steps = 2
    trainer.progress.steps_per_epoch = 2

    def compute_loss(self, batch):
        loss = self.loaded.backbone(batch["x"]).square().mean()
        if reject_first and context.rank == 1 and batch["micro"] == 0:
            loss = loss * float("nan")
        return loss, None, None

    trainer.compute_loss = MethodType(compute_loss, trainer)
    trainer.ddp = DistributedDataParallel(_TrainingGraph(trainer), broadcast_buffers=False)
    trainer._run_epoch()
    expected = torch.tensor(0.5)
    if not reject_first:
        expected -= 0.001 * 2 * expected * torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 7.0]).square().mean()
    expected -= 0.001 * 2 * expected * torch.tensor([4.0, 8.0, 9.0]).square().mean()
    torch.testing.assert_close(linear.weight.detach().reshape(()), expected, atol=1e-7, rtol=0)
    assert trainer.progress.samples_seen == 9
    assert trainer.progress.step == (1 if reject_first else 2)
    (out / f"math-rank-{context.rank}.json").write_text(
        json.dumps(
            {
                "weight": linear.weight.item(),
                "expected": expected.item(),
                "progress": trainer.progress.to_dict(),
            }
        )
    )
    trainer.emitter.close()
finally:
    dist.destroy_process_group()
