"""Generic watchdog service — routes file system events into the registering task's event loop.

Each Watchdog instance owns a single watchdog Observer that runs in its own OS thread.
Multiple paths can be registered on the same Watchdog; each registration is identified by
a unique name and can be unregistered independently. All callbacks fire in the event
loop that was active at registration time, via call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Callable, Coroutine, Dict, Optional, Set

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
)
from watchdog.observers import Observer

from peteos.utils import get_logger

_logger = get_logger(__name__)

# Maps string event names to their corresponding watchdog event classes.
_EVENT_CLASSES = {
    "created": (FileCreatedEvent, DirCreatedEvent),
    "modified": (FileModifiedEvent, DirModifiedEvent),
    "deleted": (FileDeletedEvent, DirDeletedEvent),
}


class Watchdog:
    """Owns one Observer thread and manages named path registrations.

    Args:
        name: Optional identifier for this watchdog instance, used in log output.
    """

    def __init__(self, name: Optional[str] = None):
        # Set the watchdog's display name for log messages
        self._name = name or "Watchdog"

        # Create and start the observer — it runs in its own OS thread for this Watchdog's lifetime
        self._observer = Observer()
        self._observer.start()
        _logger.debug("[%s] observer started", self._name)

        # Allocate the watchers dictionary — maps registration name to (event_loop, callback, handler, watch)
        self._watchers: Dict[str, tuple] = {}

    async def watch(
        self,
        name: str,
        path: Path,
        callback: Callable[[FileSystemEvent], None] | Callable[[FileSystemEvent], Coroutine],
        *,
        events: Optional[Set[str]] = None,
        recursive: bool = False,
    ) -> None:
        """Register a path to be watched and a callback to fire on matching events.

        The callback is invoked in the current task's event loop via
        call_soon_threadsafe. Sync callbacks run synchronously; async callbacks
        are scheduled as tasks via create_task, keeping execution cooperative.

        Args:
            name: Unique identifier for this registration on this Watchdog instance.
            path: File or directory to watch.
            callback: Called for each matching event, receives the FileSystemEvent.
                     May be a sync def or an async def — both are fully supported.
            events: Set of event type strings to filter (e.g. {"created", "deleted"}).
                     None subscribes to all event types.
            recursive: If True, watch subdirectories of path as well.

        Raises:
            ValueError: If a registration with this name already exists on this Watchdog.
        """
        # Capture the running event loop — the callback will be dispatched into this loop later
        loop = asyncio.get_running_loop()

        # Reject duplicate names on this Watchdog instance
        if name in self._watchers:
            raise ValueError(
                f"watchdog '{self._name}' already has a registration named '{name}'"
            )

        # Resolve the event filter set — None means accept all event types,
        # otherwise validate and map string names to their watchdog event classes
        event_filter: Optional[frozenset[type]]
        if events is None:
            event_filter = None
        else:
            resolved: Set[type] = set()
            for ev in events:
                if ev not in _EVENT_CLASSES:
                    raise ValueError(
                        f"unknown event type: {ev!r} (expected: created|modified|deleted)"
                    )
                resolved.update(_EVENT_CLASSES[ev])
            event_filter = frozenset(resolved)

        # Create the dispatch closure that injects the callback into the captured event loop.
        # Async callbacks are scheduled as tasks so they can be awaited; sync callbacks
        # run synchronously on the next event loop iteration.
        def _dispatch(event: FileSystemEvent) -> None:
            if asyncio.iscoroutinefunction(callback):
                loop.call_soon_threadsafe(loop.create_task, callback(event))
            else:
                loop.call_soon_threadsafe(callback, event)

        # Instantiate a per-registration handler with the dispatch closure and event filter
        handler = _Handler(dispatch=_dispatch, event_filter=event_filter)

        # Register the handler with the observer — schedule() returns the ObservedWatch for later removal
        watch = self._observer.schedule(
            handler,
            path=str(path),
            recursive=recursive,
        )

        # Store the registration so it can be found by name for later removal
        self._watchers[name] = (loop, callback, handler, watch)
        _logger.debug(
            "[%s] registered '%s' on %s (events=%s, recursive=%s)",
            self._name,
            name,
            path,
            events if events else "all",
            recursive,
        )

    def unwatch(self, name: str) -> None:
        """Remove a registered watcher and unschedule its handler from the observer.

        Args:
            name: Identifier of the registration to remove.

        Raises:
            KeyError: If no registration with this name exists on this Watchdog.
        """
        # Guard: raise if the name is not registered on this Watchdog
        if name not in self._watchers:
            raise KeyError(
                f"watchdog '{self._name}' has no registration named '{name}'"
            )

        # Pop the registration and extract the stored ObservedWatch
        _loop, _callback, _handler, watch = self._watchers.pop(name)

        # Unschedule the watch from the observer — this removes the handler from the observer entirely
        self._observer.unschedule(watch)
        _logger.debug("[%s] unregistered '%s'", self._name, name)

    def stop(self) -> None:
        """Stop the observer thread and clear all registrations.

        The observer is stopped and joined, making it safe to destroy or discard
        this Watchdog instance. After stop() the Watchdog is effectively defunct.
        """
        # Stop the observer to unblock its internal event loop
        self._observer.stop()

        # Join the observer thread to wait for it to fully terminate
        self._observer.join()

        # Discard all registrations — further unwatch() calls are now invalid
        self._watchers.clear()
        _logger.debug("[%s] observer stopped", self._name)


class _Handler(FileSystemEventHandler):
    """Per-registration handler that filters and dispatches FileSystemEvents.

    Args:
        dispatch: Callback to invoke for matching events.
        event_filter: Frozenset of event classes to accept, or None for all.
    """

    def __init__(
        self,
        dispatch: Callable[[FileSystemEvent], None],
        event_filter: Optional[frozenset[type]],
    ):
        # Store the dispatch closure for use in on_any_event
        self._dispatch = dispatch
        # Store the filter set used to drop non-matching events
        self._event_filter = event_filter

    def on_any_event(self, event: FileSystemEvent) -> None:
        # Drop events not in the filter set when one was specified
        if self._event_filter is not None and type(event) not in self._event_filter:
            return
        # Dispatch to the closure — it calls call_soon_threadsafe to route into the correct event loop
        self._dispatch(event)
