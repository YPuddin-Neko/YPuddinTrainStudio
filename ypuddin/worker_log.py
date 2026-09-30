"""Log records of the service's worker processes.

The job log view reads each record's time, level and logger from this header.
Interpreter tracebacks and Python warnings are routed through logging so they
carry the same header instead of appearing as untimed stderr text.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

log = logging.getLogger("ypuddin.worker")


def configure(level: str = "debug") -> None:
    """Record all trainer messages and other libraries from INFO or above.

    Third-party DEBUG output (image decoders, HTTP clients, compilers) would bury the
    trainer's own records, so only ``ypuddin.*`` uses the DEBUG threshold.
    """
    # Older saved configurations still pass a level; they must not hide new worker output.
    threshold = logging.DEBUG
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format=FORMAT)
    root.setLevel(max(threshold, logging.INFO))
    logging.getLogger("ypuddin").setLevel(threshold)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_exception


def _log_uncaught(kind: type[BaseException], value: BaseException, tb: Any) -> None:
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, value, tb)
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
