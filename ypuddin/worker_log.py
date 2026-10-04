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


class _EmptyFloat32Notice(logging.Filter):
    """Diffusers 0.40 warns on every ``to(dtype)``, even when no module must stay in FP32.

    That notice lists the modules it means; an empty list (``[]``) names none, so it moves to
    the debug log. A list that names modules is still a warning.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.levelno == logging.WARNING and "that should be kept in float32: []." in message:
            log.debug("%s", message)
            return False
        return True


_EMPTY_FLOAT32_NOTICE = _EmptyFloat32Notice()


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
    logging.getLogger("diffusers.models.modeling_utils").addFilter(_EMPTY_FLOAT32_NOTICE)
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
