"""CPU-only acceptance harness; product CUDA admission remains unchanged."""

import sys

import torch

from ypuddin.cli import main
from ypuddin.train.distributed import DistributedTrainer
from ypuddin.train.sharded import ShardedTrainer

# Exercise the product trainer with real CPU FSDP2, without pretending to own GPUs.
ShardedTrainer._check_capabilities = DistributedTrainer._check_capabilities
torch.cuda.max_memory_allocated = lambda *_: 0
torch.cuda.max_memory_reserved = lambda *_: 0
raise SystemExit(main(["train", sys.argv[1], "--device", "cpu"]))
