# SPDX-License-Identifier: GPL-3.0-or-later
"""Tiny pub/sub bus: worker threads publish, SSE clients on the asyncio loop consume."""

import asyncio
import threading
import time


class Bus:
    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.subs.discard(q)

    def publish(self, type: str, throttle_key: str | None = None, **data):
        """Send an event. With throttle_key, drops updates closer than 150 ms apart."""
        if throttle_key:
            now = time.monotonic()
            with self._lock:
                if now - self._last.get(throttle_key, 0) < 0.15:
                    return
                self._last[throttle_key] = now
        if not self.loop:
            return
        msg = {"type": type, **data}

        def put():
            for q in list(self.subs):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

        self.loop.call_soon_threadsafe(put)


bus = Bus()
