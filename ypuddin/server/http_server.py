"""HTTP lifecycle integration for finite shutdown of browser event streams."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable

import uvicorn

from .bus import EventBus

LAUNCHER_POLL_SECONDS = 1.0
_SYNCHRONIZE = 0x00100000
_INFINITE = 0xFFFFFFFF
_ERROR_INVALID_PARAMETER = 87  # OpenProcess: no process has this ID


class StudioServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, *, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.on_started = None

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started and self.on_started:
            self.on_started()

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Uvicorn drains HTTP connections before sending lifespan.shutdown.
        # Close SSE here for both requested restarts and console/OS shutdown.
        self.event_bus.close()
        await super().shutdown(sockets=sockets)


def watch_launcher(pid: int | None, stop: Callable[[], None]) -> bool:
    """Stop this worker the normal way once the launcher that started it is gone, so a force-quit
    launcher does not leave a worker holding the port. Returns whether a watch started.

    The launcher passes its own PID: on Windows a venv's python.exe redirector runs between the two,
    so the worker's parent is not the launcher. Windows waits on a handle opened now, which a later
    process reusing the PID cannot satisfy; POSIX polls for the worker being handed to another parent.
    """
    if not pid:
        return False

    def gone() -> None:
        stop()
        try:
            print("[studio] 启动器已退出，训练器正在关闭。", flush=True)
        except (OSError, ValueError):
            pass  # the output went to the launcher, which is gone

    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            if ctypes.get_last_error() != _ERROR_INVALID_PARAMETER:
                return False  # the launcher's handle cannot be watched; behave as an unwatched worker
            gone()  # the launcher exited before this worker started
            return True

        def wait() -> None:
            kernel.WaitForSingleObject(handle, _INFINITE)
            kernel.CloseHandle(handle)
            gone()
    else:
        child = os.getppid() == pid

        def alive() -> bool:
            if child:
                return os.getppid() == pid
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                pass
            return True

        def wait() -> None:
            while alive():
                time.sleep(LAUNCHER_POLL_SECONDS)
            gone()

    threading.Thread(target=wait, name="launcher-watch", daemon=True).start()
    return True
