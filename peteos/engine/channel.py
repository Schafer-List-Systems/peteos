"""Channel — abstract base class for connecting runners to users."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from peteos.chatbot import ContentPart, Message

if TYPE_CHECKING:
    from peteos.engine.runner import Runner


@dataclass(frozen=True)
class NotificationEvent:
    message: Message


class Channel(ABC):
    """Abstract base class for channels connecting runners to users."""

    def __init__(self) -> None:
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
    async def on_outgoing(self, message: Message) -> None:
        """Called when a message is outgoing from the runner to the external endpoint.

        Override this in your channel implementation to deliver the message
        to the external system (Nextcloud, shell, etc.).

        Args:
            message: The message to deliver.
        """
        pass

    async def _notify(self, event) -> None:
        """Internal notification handler — called by the runner.

        Unwraps the message from the event, applies filters,
        and calls on_outgoing if the message passes.
        Do not call directly — use runner.notify(channel, event) instead.
        """
        if not isinstance(event, NotificationEvent):
            return

        message = event.message
        if message is None:
            return

        # Filter out messages whose role is disabled.
        if message.role in self._disabled_roles:
            return

        # Filter out messages containing only disabled content types.
        if any(cp.type in self._disabled_content_types for cp in message.content):
            return

        # Forward to the developer's outbound hook.
        await self.on_outgoing(message)

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
