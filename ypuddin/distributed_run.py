"""Run torchrun with timestamped launcher logs before worker logging starts."""

from __future__ import annotations

import logging

from ypuddin.worker_log import FORMAT


def main(args: list[str] | None = None) -> None:
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    from torch.distributed.run import main as torchrun

    torchrun(args)


if __name__ == "__main__":
    main()
