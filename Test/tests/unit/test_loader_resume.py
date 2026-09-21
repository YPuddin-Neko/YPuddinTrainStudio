"""DataLoader recreation must preserve both worker seeds and its seed stream."""

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from ypuddin.config import TrainConfig
from ypuddin.data import BucketBatchSampler
from ypuddin.train import Trainer


class WorkerSeeds(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        worker = get_worker_info()
        return {"index": index, "worker": worker.id, "seed": worker.seed}


def make_loader_trainer(path):
    cfg = TrainConfig.model_validate({"checkpoint": {"output_dir": str(path)}, "loop": {"seed": 19}})
    trainer = Trainer(cfg, device="cpu")
    trainer.sampler = BucketBatchSampler([(64, 64)] * 8, 1, seed=19)
    trainer.loader = DataLoader(
        WorkerSeeds(), batch_sampler=trainer.sampler, num_workers=2, generator=trainer.loader_gen
    )
    return trainer


def test_resume_recreates_same_worker_seeds_without_consuming_another_epoch_seed(tmp_path):
    reference = make_loader_trainer(tmp_path / "reference")
    iterator = reference._training_iterator()
    first = next(iterator)
    reference.progress.batch_in_epoch = 1
    state = reference._capture_local_checkpoint_rng()
    # Let prefetch complete and workers terminate before starting the new pair.
    rest = list(iterator)
    original_seeds = {int(batch["worker"]): int(batch["seed"]) for batch in [first, *rest]}

    resumed = make_loader_trainer(tmp_path / "resumed")
    resumed.progress.batch_in_epoch = 1
    resumed.sampler.set_position(1)
    resumed._restore_local_checkpoint_rng(state)
    actual = list(resumed._training_iterator())
    assert [int(batch["index"]) for batch in actual] == [int(batch["index"]) for batch in rest]
    assert {int(batch["worker"]): int(batch["seed"]) for batch in actual} == original_seeds
    assert torch.equal(resumed.loader_gen.get_state(), reference.loader_gen.get_state())
    # A new epoch advances once in both runs and uses a different worker seed.
    next_seeds = []
    for trainer in (reference, resumed):
        trainer.progress.epoch = 1
        trainer.progress.batch_in_epoch = 0
        trainer.sampler.set_epoch(1)
        batches = list(trainer._training_iterator())
        next_seeds.append({int(batch["worker"]): int(batch["seed"]) for batch in batches})
    assert next_seeds[0] == next_seeds[1] != original_seeds
    assert torch.equal(resumed.loader_gen.get_state(), reference.loader_gen.get_state())


def test_legacy_resume_preserves_loader_stream_without_pre_iterator_state(tmp_path):
    trainer = make_loader_trainer(tmp_path / "legacy")
    torch.empty((), dtype=torch.int64).random_(generator=trainer.loader_gen)
    state = trainer._capture_local_checkpoint_rng()
    assert "loader_iterator" not in state
    trainer.progress.batch_in_epoch = 1
    trainer.sampler.set_position(1)
    trainer._restore_local_checkpoint_rng(state)
    before = trainer.loader_gen.get_state()
    assert len(list(trainer._training_iterator())) == 7
    assert torch.equal(trainer.loader_gen.get_state(), before)
