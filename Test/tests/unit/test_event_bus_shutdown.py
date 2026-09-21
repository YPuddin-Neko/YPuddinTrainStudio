"""Event streams finish promptly, even when a slow subscriber has filled its queue."""

import asyncio
from types import SimpleNamespace

import pytest

from ypuddin.server.bus import EventBus
from ypuddin.server.routes_core import events


@pytest.mark.asyncio
async def test_shutdown_wakes_idle_and_full_subscribers_and_rejects_new_stream_backlog():
    bus = EventBus()
    bus.attach_loop(asyncio.get_running_loop())
    idle, full = bus.subscribe(), bus.subscribe()
    for index in range(full.maxsize):
        full.put_nowait({"id": index})
    waiting = asyncio.create_task(idle.get())
    await asyncio.to_thread(bus.close)
    assert await asyncio.wait_for(waiting, 0.5) is None
    assert await asyncio.wait_for(full.get(), 0.5) is None
    assert full.empty()
    bus.close()  # Idempotent; the lifespan fallback may close it again.
    late = bus.subscribe()
    assert await asyncio.wait_for(late.get(), 0.5) is None
    bus.publish("worker.finished", {})
    await asyncio.sleep(0)
    assert idle.empty() and full.empty() and late.empty()
    bus.unsubscribe(idle)
    bus.unsubscribe(full)
    assert not bus._subs


@pytest.mark.asyncio
async def test_sse_shutdown_has_terminal_event_and_releases_subscription():
    bus = EventBus()
    bus.attach_loop(asyncio.get_running_loop())

    async def connected():
        return False

    response = await events(
        SimpleNamespace(headers={}, is_disconnected=connected), c=SimpleNamespace(bus=bus)
    )
    waiting = asyncio.create_task(anext(response.body_iterator))
    await asyncio.sleep(0)
    bus.close()
    assert await asyncio.wait_for(waiting, 0.5) == "event: service.stopping\ndata: {}\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)
    assert not bus._subs


@pytest.mark.asyncio
async def test_sse_keeps_unexpected_event_errors_visible():
    bus = EventBus()
    bus.attach_loop(asyncio.get_running_loop())

    async def connected():
        return False

    response = await events(
        SimpleNamespace(headers={}, is_disconnected=connected), c=SimpleNamespace(bus=bus)
    )
    next(iter(bus._subs)).put_nowait({"invalid": "event"})
    with pytest.raises(KeyError):
        await anext(response.body_iterator)
    assert not bus._subs
