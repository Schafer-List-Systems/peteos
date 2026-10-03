"""Channel — abstract base class for connecting runners to users."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from peteos.utils.activeclass import ActiveClass
from peteos.chatbot import ContentPart, Message

if TYPE_CHECKING:
    from peteos.engine.runner import Runner


@dataclass(frozen=True)
class NotificationEvent:
    message: Message


class Channel(ActiveClass, ABC):
    """Abstract base class for channels connecting runners to users."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._runner: "Runner | None" = None
        self._disabled_roles: set[str] = set()
        self._disabled_content_types: set[str] = set()

    async def attach(self, runner: "Runner", greeting: str | None = None) -> None:
        """Subscribe this channel to a runner.

        Args:
            runner: The runner to attach to.
            greeting: Optional greeting message to queue for the session.
        """
        # Bind the runner to this channel.
        self._runner = runner
        runner.subscribe(self)
        # Optionally greet the session on attach.
        if greeting is not None:
            msg = Message.create(
                role="assistant",
                content_parts=[ContentPart.create_text(greeting)]
            )
            await runner.queue_message(msg)

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
    async def send(self, message: Message) -> None:
        """Send a message to the user through this channel."""
        pass

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

    async def run(self) -> None:
        """Main notification consumption loop.

        Consumes events pushed via push_event() and delivers them via send().
        Runs until the channel is stopped.
        """
        # Consume and forward events until the channel stops.
        try:
            while self.is_running():
                # Wait for the next event from the queue.
                event = await self._wait()
                if event is None:
                    break
                # Unwrap the message from the event, if it is a NotificationEvent.
                if isinstance(event, NotificationEvent):
                    message = event.message
                else:
                    message = event
                if message is None:
                    break
                # Filter out messages whose role is disabled.
                if message.role in self._disabled_roles:
                    continue
                # Filter out messages containing only disabled content types.
                if any(cp.type in self._disabled_content_types for cp in message.content):
                    continue
                # Deliver the message to the user.
                await self.send(message)
        finally:
            await self.stop()
