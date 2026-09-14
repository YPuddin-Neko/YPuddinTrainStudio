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


def test_managed_restart_finishes_two_open_event_streams_before_new_worker(tmp_path):
    """Exercise the actual CLI launcher, restart API, middleware and two browser SSE connections."""
    import contextlib
    import threading

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    path = tmp_path / "managed-restart.log"
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
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as client:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    try:
                        runtime = client.get("/api/service/runtime")
                        if runtime.status_code == 200:
                            old_pid = runtime.json()["worker_id"]
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(path.read_text())
                failures, received = [], [[], []]
                with contextlib.ExitStack() as streams:
                    responses = [
                        streams.enter_context(client.stream("GET", "/api/events", timeout=4))
                        for _ in range(2)
                    ]
                    assert all(response.status_code == 200 for response in responses)

                    def consume(index, response):
                        try:
                            received[index].extend(response.iter_lines())
                        except Exception as exc:
                            failures.append(exc)

                    readers = [
                        threading.Thread(target=consume, args=(i, response), daemon=True)
                        for i, response in enumerate(responses)
                    ]
                    for reader in readers:
                        reader.start()
                    started = time.monotonic()
                    result = client.post("/api/service/restart", json={})
                    assert result.status_code == 202, result.text
                    for reader in readers:
                        reader.join(timeout=4)
                    assert all(not reader.is_alive() for reader in readers), path.read_text()
                    assert not failures, failures
                    assert time.monotonic() - started < 4
                    assert all("event: service.stopping" in lines for lines in received)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    try:
                        runtime = client.get("/api/service/runtime")
                        if runtime.status_code == 200 and runtime.json()["worker_id"] != old_pid:
                            assert runtime.json()["can_restart"]
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(path.read_text())
                assert proc.poll() is None
                content = path.read_text()
                assert "Application shutdown complete" in content
                assert "timeout graceful shutdown exceeded" not in content
                assert "Exception in ASGI application" not in content
                assert "CancelledError" not in content
                assert "Traceback" not in content
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
