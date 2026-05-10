"""Unit tests for ActiveClass base class."""

import asyncio
import pytest

from peteos.activeclass import ActiveClass


class TestableActive(ActiveClass):
    """ActiveClass subclass with a testable run() loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events_received: list = []

    async def run(self):
        while self.is_running():
            event = await self._wait()
            if event is None:
                break
            self.events_received.append(event)


class TestActiveClassInit:
    """Test ActiveClass initialization and default state."""

    def test_default_state(self):
        obj = TestableActive()
        assert obj.is_running() is False
        assert obj._loop_task is None
        assert obj._running is False
        assert obj.event_queue.empty()
        assert not obj._event_trigger.is_set()

    def test_run_raises_not_implemented(self):
        obj = ActiveClass()
        with pytest.raises(NotImplementedError):
            asyncio.run(obj.run())


class TestStartStop:
    """Test start() and stop() lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        obj = TestableActive()
        await obj.start()
        assert obj.is_running() is True
        assert obj._loop_task is not None
        assert not obj._loop_task.done()
        await obj.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        obj = TestableActive()
        await obj.start()
        await obj.stop()
        assert obj.is_running() is False
        assert obj._loop_task.done()

    @pytest.mark.asyncio
    async def test_start_already_running_raises(self):
        obj = TestableActive()
        await obj.start()
        with pytest.raises(RuntimeError, match="already running"):
            await obj.start()
        await obj.stop()

    @pytest.mark.asyncio
    async def test_stop_not_running_is_noop(self):
        obj = TestableActive()
        await obj.stop()  # should not raise
        assert obj.is_running() is False

    @pytest.mark.asyncio
    async def test_stop_twice_is_noop(self):
        obj = TestableActive()
        await obj.start()
        await obj.stop()
        await obj.stop()  # should not raise
        assert obj.is_running() is False


class TestPushEvent:
    """Test push_event() and event queue."""

    def test_push_event_adds_to_queue(self):
        obj = TestableActive()
        obj.push_event("hello")
        assert not obj.event_queue.empty()
        assert obj.event_queue.get_nowait() == "hello"

    def test_push_event_sets_trigger(self):
        obj = TestableActive()
        obj.push_event("data")
        assert obj._event_trigger.is_set()

    def test_push_event_different_types(self):
        obj = TestableActive()
        obj.push_event(42)
        obj.push_event({"key": "value"})
        obj.push_event([1, 2, 3])
        assert obj.event_queue.get_nowait() == 42
        assert obj.event_queue.get_nowait() == {"key": "value"}
        assert obj.event_queue.get_nowait() == [1, 2, 3]


class TestWait:
    """Test _wait() blocking behavior."""

    @pytest.mark.asyncio
    async def test_wait_returns_pushed_event(self):
        obj = TestableActive()

        async def push_later():
            await asyncio.sleep(0.05)
            obj.push_event("triggered")

        asyncio.create_task(push_later())
        result = await obj._wait()
        assert result == "triggered"

    @pytest.mark.asyncio
    async def test_wait_immediate_return(self):
        obj = TestableActive()
        obj.push_event("instant")
        result = await obj._wait()
        assert result == "instant"

    @pytest.mark.asyncio
    async def test_wait_timeout_returns_none(self):
        obj = TestableActive()
        result = await obj._wait(timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_blocks_without_polling(self):
        obj = TestableActive()

        async def push_later():
            await asyncio.sleep(0.05)
            obj.push_event("wakes")

        asyncio.create_task(push_later())
        result = await obj._wait()
        assert result == "wakes"


class TestSequentialEvents:
    """Test multiple sequential push_event + _wait() cycles."""

    @pytest.mark.asyncio
    async def test_sequential_events(self):
        obj = TestableActive()
        await obj.start()

        async def pusher():
            for i in range(5):
                await asyncio.sleep(0.02)
                obj.push_event(f"event_{i}")
            await obj.stop()

        asyncio.create_task(pusher())

        for i in range(5):
            result = await obj._wait()
            assert result == f"event_{i}"


class TestRunLoop:
    """Test the full run() lifecycle with events and stop."""

    @pytest.mark.asyncio
    async def test_run_loop_processes_events_and_stops(self):
        obj = TestableActive()
        await obj.start()

        async def event_provider():
            for i in range(3):
                await asyncio.sleep(0.02)
                obj.push_event(f"msg_{i}")
            await asyncio.sleep(0.02)
            await obj.stop()

        asyncio.create_task(event_provider())

        # Wait for events to be processed
        while len(obj.events_received) < 3:
            await asyncio.sleep(0.01)

        assert obj.events_received == ["msg_0", "msg_1", "msg_2"]

    @pytest.mark.asyncio
    async def test_run_with_immediate_stop(self):
        obj = TestableActive()
        await obj.start()

        async def stop_immediately():
            await asyncio.sleep(0.01)
            await obj.stop()

        asyncio.create_task(stop_immediately())

        # Wait for stop to propagate
        while obj.is_running():
            await asyncio.sleep(0.01)

        assert obj.is_running() is False
        assert len(obj.events_received) == 0


class TestConcurrency:
    """Test asyncio task safety of push_event."""

    @pytest.mark.asyncio
    async def test_concurrent_push_events(self):
        obj = TestableActive()
        num_pushes = 50

        async def pusher(i):
            await asyncio.sleep(0.001 * i)
            obj.push_event(f"concurrent_{i}")

        tasks = [asyncio.create_task(pusher(i)) for i in range(num_pushes)]
        await asyncio.gather(*tasks)

        assert obj.event_queue.qsize() == num_pushes

        retrieved = set()
        while not obj.event_queue.empty():
            retrieved.add(obj.event_queue.get_nowait())

        assert len(retrieved) == num_pushes
        for i in range(num_pushes):
            assert f"concurrent_{i}" in retrieved

    @pytest.mark.asyncio
    async def test_concurrent_wait_and_push(self):
        obj = TestableActive()
        # No start() — just test _wait() concurrency directly
        results: list = []

        async def waiter(name):
            result = await obj._wait(timeout=0.5)
            results.append((name, result))

        async def pusher():
            await asyncio.sleep(0.02)
            obj.push_event("first")
            obj.push_event("second")
            obj.push_event("third")

        asyncio.create_task(pusher())
        waiter1 = asyncio.create_task(waiter("w1"))
        waiter2 = asyncio.create_task(waiter("w2"))

        # Wait for both waiters to complete
        await waiter1
        await waiter2

        # Both waiters should have received events
        assert len(results) == 2
        values = {r[1] for r in results}
        assert values == {"first", "second"}
