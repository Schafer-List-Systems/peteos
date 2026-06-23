"""Registry for auto-discovered Message subclasses."""

from typing import Type


class MessageRegistry:
    """Registry of auto-discovered Message subclasses, keyed by class name."""

    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, message_class: Type) -> None:
        """Register a Message subclass.

        Called automatically by Message.__init_subclass__().

        Args:
            message_class: A subclass of Message.
        """
        cls._registry[message_class.__name__] = message_class

    @classmethod
    def get(cls, name: str) -> Type | None:
        """Look up a Message subclass by name.

        Args:
            name: The class name.

        Returns:
            The Message subclass, or None if not found.
        """
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> dict[str, Type]:
        """Return all registered Message subclasses.

        Returns:
            Dict mapping class name to class.
        """
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry. Useful for testing.

        Preserves the base ``Message`` class and production message subclasses
        (SystemPromptMessage, ToolDefinitionsMessage) which are auto-registered
        once via ``__init_subclass__`` and cannot be re-registered by
        re-importing cached modules.
        """
        cls._registry.clear()
        # Re-register the base Message class and all production subclasses.
        # These classes already ran __init_subclass__ during import, so
        # re-importing cached modules won't re-register them.
        from peteos.conversation.message import Message
        cls.register(Message)
        from peteos.conversation.system_prompt_message import SystemPromptMessage
        cls.register(SystemPromptMessage)
        from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage
        cls.register(ToolDefinitionsMessage)