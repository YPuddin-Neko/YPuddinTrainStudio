"""Optional metric and image sinks. Imports and remote runs only occur when explicitly configured."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig

log = logging.getLogger(__name__)


class TrainingLogs:
    def __init__(self, cfg: TrainConfig, run_dir: Path):
        log_dir = (
            Path(cfg.logging.output_dir)
            if cfg.logging.output_dir
            else Path(cfg.logging.events_path).parent
            if cfg.logging.events_path
            else run_dir
        )
        self.tensorboard: Any = None
        self.wandb: Any = None
        self.wandb_module: Any = None
        try:
            if cfg.logging.tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError as e:
                    raise ImportError("TensorBoard logging requires the tensorboard package") from e
                self.tensorboard = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
            if cfg.logging.wandb is not None:
                try:
                    import wandb
                except ImportError as e:
                    raise ImportError("W&B logging requires the wandb package") from e
                self.wandb_module = wandb
                wc = cfg.logging.wandb
                self.wandb = wandb.init(
                    project=wc.project,
                    entity=wc.entity,
                    name=wc.run_name or run_dir.name,
                    config=cfg.to_dict(),
                    dir=str(log_dir),
                )
        except Exception:
            self.close(failed=True)
            raise

    def emit(self, type_: str, data: dict[str, Any]) -> None:
        step = int(data.get("step", 0))
        values: dict[str, Any] = {}
        if type_ == "step":
            values = {
                f"train/{k}": data[k]
                for k in ("loss", "loss_ema", "grad_norm", "it_s", "vram_mb")
                if data.get(k) is not None
            }
            values.update({f"lr/{k}": v for k, v in data.get("lr", {}).items()})
        elif type_ == "validation":
            values = {"validation/mean": data["mean"]}
            values.update({f"validation/t_{k}": v for k, v in data.get("per_t", {}).items()})
        for tag, value in values.items():
            if self.tensorboard is not None:
                self.tensorboard.add_scalar(tag, value, global_step=step)
        if type_ == "sample.saved":
            tag = f"samples/{data.get('prompt_index', 0)}"
            if self.tensorboard is not None:
                import numpy as np
                from PIL import Image

                with Image.open(data["path"]) as im:
                    self.tensorboard.add_image(tag, np.asarray(im.convert("RGB")), step, dataformats="HWC")
            if self.wandb is not None:
                values[tag] = self.wandb_module.Image(data["path"], caption=data.get("prompt", ""))
        if values and self.wandb is not None:
            self.wandb.log(values, step=step)

    def close(self, *, failed: bool = False) -> None:
        for sink, action in ((self.tensorboard, "close"), (self.wandb, "finish")):
            if sink is not None:
                try:
                    getattr(sink, action)(**({"exit_code": 1 if failed else 0} if action == "finish" else {}))
                except Exception:
                    log.exception("could not close training log sink")
        self.tensorboard = self.wandb = None
