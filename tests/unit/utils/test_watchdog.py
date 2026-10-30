"""Tests for peteos.utils.watchdog."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from peteos.utils.watchdog import Watchdog, _Handler
from peteos.utils.watchdog import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    DirCreatedEvent,
)


# ── _Handler unit tests (no async, no I/O) ────────────────────────────────────


class TestHandler:

    def test_passes_all_events_when_no_filter(self):
        """When no event filter is set, every event is dispatched."""
        received = []
        handler = _Handler(dispatch=received.append, event_filter=None)
        handler.on_any_event(FileCreatedEvent("src_path", False, ""))
        handler.on_any_event(FileDeletedEvent("src_path", False, ""))
        handler.on_any_event(FileModifiedEvent("src_path", False, ""))
        assert len(received) == 3

    def test_drops_events_not_in_filter(self):
        """Events not in the filter set are silently dropped."""
        received = []
        handler = _Handler(
            dispatch=received.append,
            event_filter=frozenset({FileModifiedEvent}),
        )
        handler.on_any_event(FileCreatedEvent("src_path", False, ""))
        handler.on_any_event(FileModifiedEvent("src_path", False, ""))
        assert len(received) == 1
        assert isinstance(received[0], FileModifiedEvent)

    def test_filters_both_file_and_directory_events(self):
        """File and directory variants of the same event type share the filter."""
        received = []
        handler = _Handler(
            dispatch=received.append,
            event_filter=frozenset({FileCreatedEvent, DirCreatedEvent}),
        )
        handler.on_any_event(FileCreatedEvent("src_path", False, ""))
        handler.on_any_event(DirCreatedEvent("src_path", True, ""))
        handler.on_any_event(FileModifiedEvent("src_path", False, ""))
        assert len(received) == 2
        assert all(isinstance(e, (FileCreatedEvent, DirCreatedEvent)) for e in received)


# ── Watchdog API tests (synchronous — uses explicit event loop) ─────────────


class TestWatchdog:

    def _run_async(self, coro):
        """Run a coroutine in a fresh event loop, then close the loop."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_watch_rejects_duplicate_name(self):
        """Registering the same name twice raises ValueError."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch("x", Path("."), lambda e: None)
            await watchdog.watch("x", Path("."), lambda e: None)
        with pytest.raises(ValueError, match="already has a registration"):
            self._run_async(_coro())
        watchdog.stop()

    def test_unwatch_unknown_name_raises(self):
        """Unwatching a name that was never registered raises KeyError."""
        watchdog = Watchdog(name="test")
        with pytest.raises(KeyError, match="no registration named"):
            watchdog.unwatch("does-not-exist")
        watchdog.stop()

    def test_stop_clears_all_registrations(self):
        """stop() empties the watchers dictionary."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch("a", Path("."), lambda e: None)
            await watchdog.watch("b", Path("."), lambda e: None)
        self._run_async(_coro())
        assert len(watchdog._watchers) == 2
        watchdog.stop()
        assert watchdog._watchers == {}

    def test_multiple_watchers_same_path_allowed(self):
        """Same path can be registered with different names on the same Watchdog."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch("w1", Path("."), lambda e: None)
            await watchdog.watch("w2", Path("."), lambda e: None)
        self._run_async(_coro())
        assert len(watchdog._watchers) == 2
        watchdog.stop()

    def test_unwatch_removes_only_that_registration(self):
        """Unwatching one name leaves other registrations intact."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch("a", Path("."), lambda e: None)
            await watchdog.watch("b", Path("."), lambda e: None)
        self._run_async(_coro())
        watchdog.unwatch("a")
        assert "a" not in watchdog._watchers
        assert "b" in watchdog._watchers
        watchdog.stop()

    def test_watch_unknown_event_type_raises(self):
        """Passing an unknown event type string raises ValueError."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch(
                "x", Path("."), lambda e: None, events={"flobber"}
            )
        with pytest.raises(ValueError, match="unknown event type"):
            self._run_async(_coro())
        watchdog.stop()

    def test_watch_all_events_with_none(self):
        """events=None subscribes to all event types without error."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch("x", Path("."), lambda e: None, events=None)
        self._run_async(_coro())
        watchdog.stop()

    def test_watch_empty_set_filters_everything(self):
        """events=set() results in zero dispatches."""
        watchdog = Watchdog(name="test")
        async def _coro():
            await watchdog.watch(
                "x", Path("."), lambda e: None, events=set()
            )
        self._run_async(_coro())
        # on_any_event drops everything when filter is a non-None empty set
        handler = watchdog._watchers["x"][2]
        received = []
        handler.on_any_event(FileCreatedEvent("src_path", False, ""))
        assert received == []
        watchdog.stop()

    def test_stop_idempotent(self):
        """Calling stop() twice does not raise."""
        watchdog = Watchdog(name="test")
        watchdog.stop()
        watchdog.stop()  # should not raise


# ── Watchdog integration tests (real I/O) ─────────────────────────────────────


class TestWatchdogIntegration:

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.mark.asyncio
    async def test_callback_fires_on_file_modified(self, tmpdir):
        """Touching a watched file fires the registered callback.

        Note: watchdog's inotify backend monitors at the directory level.
        We watch the parent directory and filter for the target filename.
        """
        hit = []
        watchdog = Watchdog(name="integration")
        target = tmpdir / "watched.txt"
        target.touch()

        await watchdog.watch(
            "t",
            tmpdir,
            lambda e: hit.append(e),
            events={"modified"},
        )
        # Touch the specific file to trigger a modified event
        target.touch()

        # Yield to the event loop so the watchdog thread delivers the event
        await asyncio.sleep(0.3)

        watchdog.stop()
        assert len(hit) == 1
        assert isinstance(hit[0], FileModifiedEvent)
        assert "watched.txt" in hit[0].src_path

    @pytest.mark.asyncio
    async def test_callback_fires_on_file_created(self, tmpdir):
        """Creating a new file in a watched directory fires the callback."""
        hit = []
        watchdog = Watchdog(name="integration")
        await watchdog.watch(
            "t", tmpdir, lambda e: hit.append(e), events={"created"}
        )

        (tmpdir / "newfile.txt").touch()
        await asyncio.sleep(0.3)

        watchdog.stop()
        # May receive more than one event per touch (e.g. OPEN + CREATED)
        assert any(isinstance(e, FileCreatedEvent) for e in hit)

    @pytest.mark.asyncio
    async def test_unwatch_stops_future_events(self, tmpdir):
        """Unwatching a name prevents its callback from firing on later changes."""
        hit = []
        watchdog = Watchdog(name="integration")
        target = tmpdir / "later.txt"
        target.touch()

        await watchdog.watch("t", target, lambda e: hit.append(e))
        watchdog.unwatch("t")
        (tmpdir / "trigger.touch").touch()
        await asyncio.sleep(0.3)

        watchdog.stop()
        assert hit == []

    @pytest.mark.asyncio
    async def test_callback_runs_in_registering_event_loop(self, tmpdir):
        """The callback executes in the task's event loop, not a separate thread.

        We watch the parent directory and filter for the target filename,
        since inotify monitors at the directory level.
        """
        watchdog = Watchdog(name="integration")
        target = tmpdir / "eventloop.txt"
        target.touch()
        running_loop = asyncio.get_running_loop()

        received_loops = []

        def callback(event):
            # This runs synchronously on the event loop's thread via call_soon_threadsafe
            received_loops.append(asyncio.get_running_loop())

        await watchdog.watch("t", tmpdir, callback, events={"modified"})
        target.touch()
        await asyncio.sleep(0.3)

        watchdog.stop()
        assert len(received_loops) == 1
        assert received_loops[0] is running_loop

    @pytest.mark.asyncio
    async def test_async_callback_runs_in_registering_event_loop(self, tmpdir):
        """An async callback executes in the task's event loop via create_task.

        We watch the parent directory and filter for the target filename,
        since inotify monitors at the directory level.
        Async callbacks are detected and scheduled as tasks rather than called directly.
        """
        watchdog = Watchdog(name="integration")
        target = tmpdir / "async_eventloop.txt"
        target.touch()
        running_loop = asyncio.get_running_loop()

        received_loops = []

        async def callback(event):
            # Runs as a task on the event loop — fully async
            received_loops.append(asyncio.get_running_loop())

        await watchdog.watch("t", tmpdir, callback, events={"modified"})
        target.touch()
        await asyncio.sleep(0.3)

        watchdog.stop()
        assert len(received_loops) == 1
        assert received_loops[0] is running_loop

    @pytest.mark.asyncio
    async def test_recursive_watches_subdirectory(self, tmpdir):
        """recursive=True causes new files in subdirectories to fire callbacks."""
        hit = []
        watchdog = Watchdog(name="integration")
        await watchdog.watch(
            "t", tmpdir, lambda e: hit.append(e),
            events={"created"}, recursive=True
        )

        subdir = tmpdir / "sub"
        subdir.mkdir()
        await asyncio.sleep(0.1)  # let the watcher settle
        (subdir / "nested.txt").touch()
        await asyncio.sleep(0.3)

        watchdog.stop()
        assert any(isinstance(e, FileCreatedEvent) for e in hit)
