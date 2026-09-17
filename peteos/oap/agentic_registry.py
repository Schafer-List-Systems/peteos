"""Registry for auto-discovered AgenticObject subclasses."""

import copy
import importlib
import pkgutil
from typing import Type

from peteos.persona.role import Role


def _is_oap_object(cls: type) -> bool:
    """Check whether a class directly inherits from AgenticObject or AdaptiveObject.

    Returns True for any class whose direct bases include AgenticObject or AdaptiveObject.
    These are the classes in the MRO whose docstrings and config should be collected.
    """
    from peteos.oap.agentic_object import AgenticObject
    from peteos.oap.adaptive_object import AdaptiveObject
    return AgenticObject in cls.__bases__ or AdaptiveObject in cls.__bases__


def _build_system_prompt(cls: type) -> str:
    """Build the system prompt for a concrete agentic object class.

    Collects docstrings from classes in the MRO that have _oap_config
    (set by @agentic_object decorator), ordered from base to most-derived.
    Prepends standard behaviour directives.
    """
    class_name = cls.__name__

    doc_parts: list[str] = []
    for c in reversed(cls.__mro__):
        if not _is_oap_object(c):
            continue
        c_doc = (c.__doc__ or "").strip()
        if c_doc:
            doc_parts.append(c_doc)

    if doc_parts:
        return "\n\n".join(doc_parts)
    return (
        f"You are an agent working on a {class_name} object. "
        f"You have tools to read and modify the state of this object. "
        f"Use those tools and the information you already have to fulfill the user's request. "
        f"NEVER ask the user for more information or clarification. "
        f"If you cannot produce the requested output, use the `produce_error` to return an error message explaining why. "
    )


class AgenticObjectRegistry:
    """Registry of auto-discovered AgenticObject subclasses, keyed by class name."""

    _registry: dict[str, Type] = {}
    _role_cache: dict[str, Role] = {}

    @classmethod
    def create_role(cls, cls_name: str) -> Role:
        """Create a mutable Role copy for an AgenticObject class by name.

        Builds the canonical Role (loaded from disk if role_config, else
        from defaults), merges with any user-provided Role from the
        RoleManager singleton (model, description, system_prompt hooks),
        and returns a copy so each instance can mutate independently.
        """
        agentic_class = cls._registry.get(cls_name)
        if not agentic_class:
            raise ValueError(f"AgenticObject class '{cls_name}' not found in registry")

        # Build or load the canonical role.
        canonical = cls._role_cache.get(cls_name) or cls._resolve_canonical_role(
            cls_name, agentic_class
        )

        # Merge with user-provided Role from RoleManager singleton.
        user_role = cls._get_user_role(canonical.name)
        role = copy.copy(canonical)
        role = cls._merge_user_overrides(role, user_role)

        return role

    @classmethod
    def _get_user_role(cls, name: str) -> Role | None:
        """Look up a user-provided Role in the RoleManager class."""
        from peteos.persona.rolemanager import RoleManager
        return RoleManager.get_role(name)

    @classmethod
    def _merge_user_overrides(cls, role: Role, user_role: Role | None) -> Role:
        """Merge user-provided Role fields into the canonical Role.

        User can override: model, description.
        User can extend: system_prompt_hooks (appended).
        Everything else is owned by the AO registry.
        """
        if user_role is None:
            return role

        if user_role.model:
            role.model = user_role.model
        if user_role.description:
            role.description = user_role.description
        if user_role.system_prompt:
            role.system_prompt_hooks.append(lambda sp=user_role.system_prompt: sp)
        if user_role.system_prompt_hooks:
            role.system_prompt_hooks.extend(user_role.system_prompt_hooks)
        return role

    @classmethod
    def _resolve_canonical_role(cls, cls_name: str, agentic_class: Type) -> Role:
        """Build the canonical Role for the given class."""
        role_name = getattr(agentic_class, "_oap_config", {}).get("role") or cls_name
        canonical = Role(
            name=role_name,
            description=f"Agent for {cls_name}",
            system_prompt=_build_system_prompt(agentic_class),
            model=".*",
            tool_filter=[".*"],
        )

        cls._role_cache[cls_name] = canonical
        return canonical

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