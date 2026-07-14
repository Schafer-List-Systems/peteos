"""Scope class for hierarchical namespace management."""

from __future__ import annotations


class Scope:
    """A hierarchical namespace with a parent reference."""

    def __init__(self, name: str, parent: Scope | None = None) -> None:
        self.name = name
        self.parent = parent
        self.entries: dict[str, object] = {}

    def __getattr__(self, name: str) -> object:
        if name in self.entries:
            return self.entries[name]
        if self.parent is not None:
            return getattr(self.parent, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
