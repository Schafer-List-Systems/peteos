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
        """Clear the registry. Useful for testing."""
        cls._registry.clear()