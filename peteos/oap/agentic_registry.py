"""Registry for auto-discovered AgenticObject subclasses."""

import importlib
import pkgutil
from typing import Type


class AgenticObjectRegistry:
    """Registry of auto-discovered AgenticObject subclasses, keyed by class name."""

    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, agentic_class: Type) -> None:
        """Register an AgenticObject subclass.

        Called automatically by AgenticObject.__init_subclass__().

        Args:
            agentic_class: A subclass of AgenticObject.
        """
        cls._registry[agentic_class.__name__] = agentic_class

    @classmethod
    def get(cls, name: str) -> Type | None:
        """Look up an AgenticObject subclass by name.

        Args:
            name: The class name.

        Returns:
            The AgenticObject subclass, or None if not found.
        """
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> dict[str, Type]:
        """Return all registered AgenticObject subclasses.

        Returns:
            Dict mapping class name to class.
        """
        return dict(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry. Useful for testing."""
        cls._registry.clear()

    @classmethod
    def discover(
        cls,
        base_package: str = "peteos",
        sub_package: str | None = None,
    ) -> int:
        """Discover and register AgenticObject subclasses by importing modules.

        Walks the specified package, imports each .py module, and lets
        AgenticObject.__init_subclass__ handle auto-registration.

        Args:
            base_package: The top-level package to scan (default: "peteos").
            sub_package: A sub-package to scan within base_package.
                         If None, scans the base package directly.

        Returns:
            Number of modules scanned.
        """
        if sub_package is not None:
            package_path = f"{base_package}.{sub_package}"
        else:
            package_path = base_package

        count = 0
        try:
            package = importlib.import_module(package_path)
        except ModuleNotFoundError:
            return 0

        if not hasattr(package, "__path__"):
            return count

        for finder, module_name, ispkg in pkgutil.walk_packages(
            path=package.__path__,
            prefix=f"{package_path}.",
            onerror=lambda name: None,
        ):
            try:
                importlib.import_module(module_name)
                count += 1
            except Exception:
                pass

        return count