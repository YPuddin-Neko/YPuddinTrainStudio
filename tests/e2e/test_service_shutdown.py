"""An open browser event stream must not prevent CLI shutdown from reaching lifespan cleanup."""

import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT lifecycle; Windows has a separate console contract")
def test_cli_shutdown_with_open_sse(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    path = tmp_path / "server.log"
    with path.open("wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ypuddin.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--data-root",
                str(tmp_path / "studio"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2) as client:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    raise AssertionError(path.read_text())
                with client.stream("GET", "/api/events", timeout=None) as response:
                    assert response.status_code == 200
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=12)
            assert "Application shutdown complete" in path.read_text()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
