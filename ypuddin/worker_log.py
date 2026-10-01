"""Log records of the service's worker processes.

The job log view reads each record's time, level and logger from this header.
Interpreter tracebacks and Python warnings are routed through logging so they
carry the same header instead of appearing as untimed stderr text.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

log = logging.getLogger("ypuddin.worker")


def _worker_format() -> str:
    try:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError):
        return FORMAT
    return f"[rank{rank}]: {FORMAT}" if world_size > 1 and 0 <= rank < world_size else FORMAT


def configure() -> None:
    """Record all trainer messages and other libraries from INFO or above.

    Third-party DEBUG output (image decoders, HTTP clients, compilers) would bury the
    trainer's own records, so only ``ypuddin.*`` uses the DEBUG threshold.
    """
    root = logging.getLogger()
    format_ = _worker_format()
    if not root.handlers:
        logging.basicConfig(format=format_)
    for handler in root.handlers:
        # The CLI installs its standard stream handler before it knows this is a worker.
        if type(handler) is logging.StreamHandler:
            handler.setFormatter(logging.Formatter(format_))
    root.setLevel(logging.INFO)
    logging.getLogger("ypuddin").setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_exception


def mark_logged(error: BaseException) -> None:
    """The worker has written this exception's traceback; the exit hook must not repeat it."""
    try:
        error._ypuddin_logged = True
    except AttributeError:
        pass


def _log_uncaught(kind: type[BaseException], value: BaseException, tb: Any) -> None:
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, value, tb)
        return
    if getattr(value, "_ypuddin_logged", False):
        return
    log.error("worker stopped by an unhandled exception", exc_info=(kind, value, tb))


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread else "unknown"
    log.error(
        "thread %s stopped by an unhandled exception",
        name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
