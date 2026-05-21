import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable
from enum import Enum

from peteos.chatbot import ChatBot, ChatBotManager, ChatHistory
from peteos.role import Role
from peteos.toolmanager import ToolManager


class ExecStatus(str, Enum):
    FINISHED = "finished"
    INTERRUPTED = "interrupted"
    CONTINUE = "continue"
    PENDING = "pending"
    ERROR = "error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_FAILED = "tool_failed"
    TOOL_DENIED = "tool_denied"


class ExecutionEnvironment(ABC):
    """Environment for executing agent actions."""

    def __init__(
        self,
        chatbot_manager: ChatBotManager,
        chat_history: ChatHistory,
        tool_manager: ToolManager,
        role: Role
    ):
        """
        Initialize ExecutionEnvironment.

        Args:
            chatbot_manager: The ChatBotManager instance to use.
            chat_history: The ChatHistory instance to use.
            tool_manager: The ToolManager instance to use.
            role: The Role instance to use (for model selection).
        """
        self._chatbot: ChatBot = ExecutionEnvironment._select_chatbot(chatbot_manager, role)
        self._interrupt = False
        self._completion_signal: asyncio.Event = asyncio.Event()
        self._completion_signal.set()  # Start as signaled (not running)
        self._hooks: dict[str, list[Callable]] = {
            "before_tool_execution": [],
            "after_tool_execution": [],
            "before_notification_publish": [],
            "before_send_to_chatbot": [],
        }

    @property
    def chatbot(self) -> ChatBot:
        """Get current ChatBot, selecting from manager if available."""
        return self._chatbot

    @staticmethod
    def _select_chatbot(chatbot_manager: ChatBotManager, role: Role) -> ChatBot:
        """Select a ChatBot from the manager based on role.model."""
        chatbots = chatbot_manager.list_chatbots(role.model)
        if not chatbots:
            raise ValueError(
                f"No ChatBot found matching model pattern '{role.model}' "
                f"for role '{role.name}'"
            )
        return chatbots[0][1]

    @property
    def is_running(self) -> bool:
        """Check if the execution environment is currently running."""
        return not self._completion_signal.is_set()

    def set_interrupt(self) -> None:
        """Request interruption of the execution loop."""
        self._interrupt = True

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        self._interrupt = False

    def get_chat_history(self) -> ChatHistory:
        """Get the internal chat history."""
        return self.chat_history

    async def step(self, session: "Session") -> tuple[ExecStatus, dict | None]:
        """Execute one loop iteration: chatbot call → tool(s) → continue/exit.

        Args:
            session: The Session owning the state consumed by this step.

        Returns:
            Tuple of (status, data).
        """
        raise NotImplementedError

    async def wait_for_stop(self) -> None:
        """
        Wait for the execution loop to complete.

        This blocks until the loop finishes (either naturally or via interruption).
        """
        await self._completion_signal.wait()

    def register_hook(self, hook_point: str, callback: Callable, *args: Any) -> None:
        """Register a hook callback for a specific hook point.

        Args:
            hook_point: "before_tool_execution", "after_tool_execution",
                or "before_notification_publish".
            callback: The hook function to register. Can be sync or async.
            *args: Additional arguments to pass to the callback when called.
                These will be prepended to any arguments passed at call time.

        Raises:
            ValueError: If hook_point is not a valid hook point.
        """
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        # Use partial to bind extra args to the callback
        self._hooks[hook_point].append(partial(callback, *args))

    def deregister_hook(self, hook_point: str, callback: Callable) -> None:
        """Deregister a specific hook callback.

        Args:
            hook_point: "before_tool_execution", "after_tool_execution",
                "before_notification_publish", or "before_send_to_chatbot".
            callback: The hook function to remove.

        Raises:
            ValueError: If hook_point is not a valid hook point.
            ValueError: If callback is not registered for the hook point.
        """
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        # Remove by checking the func attribute (for partial functions)
        for hook in list(self._hooks[hook_point]):
            # Check if it's a partial wrapping the original callback
            if hasattr(hook, 'func') and hook.func == callback:
                self._hooks[hook_point].remove(hook)
                return
            elif hook == callback:
                self._hooks[hook_point].remove(hook)
                return
        raise ValueError(f"Callback not found for hook point '{hook_point}'")

    def deregister_all_hooks(self, hook_point: str) -> None:
        """Deregister all hooks for a specific hook point.

        Args:
            hook_point: "before_tool_execution", "after_tool_execution",
                "before_notification_publish", or "before_send_to_chatbot".

        Raises:
            ValueError: If hook_point is not a valid hook point.
        """
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")
        self._hooks[hook_point].clear()

    def _call_hooks(self, hook_point: str, *args: Any) -> Any | None:
        """Call all hooks registered for a specific hook point.

        For `before_tool_execution`, returns the first non-None result from hooks,
        which can be a tuple (allow: bool, message: str) to disallow the tool call.

        Args:
            hook_point: "before_tool_execution", "after_tool_execution",
                "before_notification_publish", or "before_send_to_chatbot".
            *args: Arguments to pass to each hook callback.

        Returns:
            The return value from the first hook that returns a value, or None.

        Raises:
            ValueError: If hook_point is not a valid hook point.
        """
        if hook_point not in self._hooks:
            raise ValueError(f"Unknown hook point: {hook_point}")

        for callback in self._hooks[hook_point]:
            result = callback(*args)
            if result is not None:
                return result
        return None
