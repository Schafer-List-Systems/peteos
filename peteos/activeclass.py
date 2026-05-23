"""ActiveClass - Base class for components that run their own event loop."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

_logger = logging.getLogger(__name__)


class ActiveClass:
    """Base class for components with their own event loop.

    Provides lifecycle management (start/stop/is_running), an event queue
    for incoming events, and a non-blocking _wait() method that subclasses
    use to idle without polling in their run() loop.

    Example usage::

        class MyComponent(ActiveClass):
            async def run(self):
                while self.is_running():
                    event = await self._wait()
                    if event is None:
                        break
                    self.handle_event(event)

    Call ``await component.start()`` to begin the background loop,
    and ``await component.stop()`` to end it.
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._event_trigger: asyncio.Event = asyncio.Event()
        self.event_queue: asyncio.Queue[Any] = asyncio.Queue()

    def is_running(self) -> bool:
        """Check if the component is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the background task that runs self.run().

        Raises:
            RuntimeError: If already running.
        """
        if self._running:
            raise RuntimeError("ActiveClass is already running")

        self._running = True
        self._loop_task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Stop the background task gracefully.

        No-op if not running.
        """
        if not self._running:
            return

        self._running = False

        # When called from within the running task itself (e.g. exception unwind
        # in _main_loop), we can't await the current task — that's a deadlock.
        # Just clean up state and return.
        if asyncio.current_task() is self._loop_task:
            self._loop_task = None
            return

        # Send sentinel to unblock _wait() so it can see _running is False.
        # Without this, stop() cancelling the task could steal the wakeup
        # before _wait() has a chance to drain the queue.
        self.event_queue.put_nowait(None)
        self._event_trigger.set()

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _main_loop(self) -> None:
        """Internal loop that runs self.run() and handles cancellation."""
        try:
            await self.run()
        except Exception as e:
            _logger.error("[%s]: _main_loop: exception raised: %s: %r", type(self).__name__, type(e).__name__, e)
        finally:
            await self.stop()

    async def _wait_for_event(self, timeout: Optional[float] = None) -> bool:
        """Block until an event arrives in the queue.

        Fast-path: if an event is already queued, returns True immediately.
        Otherwise waits on asyncio.Event for wake-up signal.

        Uses a retry loop to handle spurious wakeups (multiple waiters
        waking on one trigger): if the queue is empty after wakeup, wait
        again with a reduced timeout.

        Does NOT consume the event from the queue.

        Returns:
            True if an event is available, False if timeout expired.
        """
        if not self.event_queue.empty():
            _logger.debug("[ActiveClass:%s] event already in queue, returning True immediately", type(self).__name__)
            return True

        _logger.debug("[ActiveClass:%s] no event, starting wait", type(self).__name__)
        self._event_trigger.clear()
        deadline = (asyncio.get_event_loop().time() + timeout) if timeout else None

        while True:
            remaining = (deadline - asyncio.get_event_loop().time()) if deadline else None
            try:
                await asyncio.wait_for(
                    self._event_trigger.wait(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return False

            if not self.event_queue.empty():
                _logger.debug("[ActiveClass:%s] received event after wait, returning True", type(self).__name__)
                return True

    async def _wait(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Block until an event arrives, consume it from the queue, and return it."""
        if await self._wait_for_event(timeout):
            return self.event_queue.get_nowait()
        return None

    async def run(self) -> None:
        """Main loop body to be implemented by subclasses."""
        raise NotImplementedError

    def has_event(self) -> bool:
        """Check if there is an event available in the queue (non-blocking)."""
        return not self.event_queue.empty()

    def push_event(self, event: Any) -> None:
        """Push an event into the event queue and wake any _wait()."""
        self.event_queue.put_nowait(event)
        self._event_trigger.set()
        _logger.debug("[ActiveClass:%s] pushed event (%s), queue_size=%d, trigger set", type(self).__name__, event.__class__.__name__ if event is not None else "None", self.event_queue.qsize())
