"""Channel — abstract base class for connecting runners to users."""

from __future__ import annotations

import uuid as _uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from peteos.chatbot import ContentPart, Message
from peteos.engine.approval_voting import ApprovalDecision
from peteos.engine.events import ActivityEvent, ActivityState, ApprovalEvent, MessageEvent, ToolExecutionEvent

if TYPE_CHECKING:
    from peteos.engine.runner import Runner


class Channel(ABC):
    """Abstract base class for channels connecting runners to users."""

    def __init__(self) -> None:
        self._id: str = str(_uuid.uuid4())
        self._runner: "Runner | None" = None
        self._disabled_roles: set[str] = set()
        self._disabled_content_types: set[str] = set()
        self._tool_calls: dict[str, ContentPart] = {}

    async def attach(self, runner: "Runner", greeting: str | None = None) -> None:
        """Subscribe this channel to a runner.

        Args:
            runner: The runner to attach to.
            greeting: Optional greeting message to queue for the session.
        """
        # Bind the runner to this channel and announce current activity state.
        self._runner = runner
        runner.subscribe(self)
        await self._notify(ActivityEvent(state=runner._activity_state))

        # Optionally greet the session on attach.
        if greeting is not None:
            msg = Message.create(
                role="assistant",
                content_parts=[ContentPart.create_text(greeting)]
            )
            await runner.queue_message(msg)

    def get_activity_state(self) -> "ActivityState":
        """Return the runner's current activity state."""
        # Precondition: ensure the channel is attached to a runner.
        if self._runner is None:
            raise RuntimeError("Channel is not attached to a runner")

        return self._runner._activity_state

    def get_pending_tool_calls(self) -> list:
        """Return pending tool call records from the execution environment."""
        # Precondition: ensure the channel is attached to a runner.
        if self._runner is None:
            raise RuntimeError("Channel is not attached to a runner")

        return self._runner._execution_environment.get_pending_tool_calls()

    def approve_tool(self, call_id: str) -> None:
        """Vote to approve a pending tool call.

        Args:
            call_id: The tool call ID to approve.
        """
        # Precondition: ensure the channel is attached to a runner.
        if self._runner is None:
            raise RuntimeError("Channel is not attached to a runner")

        # Resolve the election for this call_id from the execution environment.
        election = self._runner._execution_environment._elections.get(call_id)
        if election is None:
            return

        # Cast the channel's vote as approved into the election.
        election.vote(self._id, ApprovalDecision.APPROVED)

    def deny_tool(self, call_id: str, reason: str | None = None) -> None:
        """Vote to deny a pending tool call.

        Args:
            call_id: The tool call ID to deny.
            reason: Optional denial reason.
        """
        # Precondition: ensure the channel is attached to a runner.
        if self._runner is None:
            raise RuntimeError("Channel is not attached to a runner")

        # Resolve the election for this call_id from the execution environment.
        election = self._runner._execution_environment._elections.get(call_id)
        if election is None:
            return

        # Cast the channel's vote as denied into the election.
        election.vote(self._id, ApprovalDecision.DENIED, reason)

    async def detach(self, farewell: str | None = None) -> None:
        """Unsubscribe this channel from its runner.

        Args:
            farewell: Optional farewell message to queue for the session.
        """
        # Unbind from the runner, if still attached.
        if self._runner is not None:

            # Optionally bid farewell to the session on detach.
            if farewell is not None:
                msg = Message.create(
                    role="assistant",
                    content_parts=[ContentPart.create_text(farewell)]
                )
                await self._runner.queue_message(msg)

            # Unsubscribe and clear the runner reference.
            self._runner.unsubscribe(self)
            self._runner = None

    @abstractmethod
    async def on_outgoing(self, event: MessageEvent | ActivityEvent | ApprovalEvent | ToolExecutionEvent) -> None:
        """Called when a message or domain event is outgoing from the runner.

        Override this in your channel implementation to deliver the event
        to the external system (Nextcloud, shell, etc.).

        Args:
            event: A MessageEvent or a domain event (ActivityEvent,
                ApprovalEvent, ToolExecutionEvent) to deliver.
        """
        pass

    async def _notify(self, event: MessageEvent | ActivityEvent | ApprovalEvent | ToolExecutionEvent) -> None:
        """Internal notification handler — called by the runner.

        For MessageEvent: applies role and content-type filters before delivery.
        For domain events: delivered directly to on_outgoing without filtering.
        For unknown types: raises TypeError.

        Do not call directly — use runner._notify_channels(event) instead.
        """
        if isinstance(event, MessageEvent):
            message = event.message
            if message is None:
                return
            if message.role in self._disabled_roles:
                return
            if any(cp.type in self._disabled_content_types for cp in message.content):
                return

            # Register pending tool_use parts so approve/deny can resolve call_ids later.
            for cp in message.content:
                if cp.type == "tool_use" and cp.call_id:
                    self._tool_calls[cp.call_id] = cp

        if not isinstance(event, (MessageEvent, ActivityEvent, ApprovalEvent, ToolExecutionEvent)):
            raise TypeError(f"Unknown event type {type(event).__name__}: {event!r}")

        await self.on_outgoing(event)

    async def enqueue(self, message: Message) -> None:
        """Enqueue a message into the runner from an external source.

        Called by the channel implementer when an external event arrives
        (e.g. a webhook from Nextcloud) — injects the message into the
        runner's processing queue.

        Args:
            message: The message to enqueue.

        Raises:
            RuntimeError: Channel is not attached to a runner.
        """
        # Inject the message into the runner's queue.
        if self._runner is None:
            raise RuntimeError("Channel is not attached to a runner")

        await self._runner.queue_message(message)

    def enable_role(self, role: str, on: bool) -> None:
        """Enable or disable a message role for this channel.

        Args:
            role: The message role to control (e.g. "reasoning", "tool", "tool_result").
            on: True to enable, False to disable.
        """
        # Control whether a message role passes through this channel.
        if on:
            self._disabled_roles.discard(role)
        else:
            self._disabled_roles.add(role)

    def enable_content_type(self, content_type: str, on: bool) -> None:
        """Enable or disable a content part type for this channel.

        Args:
            content_type: The content part type to control (e.g. "text", "thinking", "tool_use").
            on: True to enable, False to disable.
        """
        # Control whether a content part type passes through this channel.
        if on:
            self._disabled_content_types.discard(content_type)
        else:
            self._disabled_content_types.add(content_type)
