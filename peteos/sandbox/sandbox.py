"""Sandbox class for restricted code execution."""

from __future__ import annotations

import contextvars
from typing import Any

from peteos.sandbox.scope import Scope


class Sandbox:
    """A sandbox object where attribute resolution follows the caller's namespace."""

    _calling_ns: contextvars.ContextVar[Scope | None] = contextvars.ContextVar(
        "_calling_ns", default=None,
    )

    def __init__(self, scope: Scope) -> None:
        self._scope = scope

    def __getattr__(self, name: str) -> object:
        """Resolve attribute via the caller's scope, then walk up the parent chain."""
        caller_ns = Sandbox._calling_ns.get()
        ns = caller_ns if caller_ns is not None else self._scope
        obj = getattr(ns, name)
        if callable(obj):
            return lambda *args, **kwargs: obj(self, *args, **kwargs)
        return obj

    def __repr__(self) -> str:
        names: list[str] = []
        ns: Scope | None = self._scope
        while ns is not None:
            names.append(ns.name)
            ns = ns.parent
        return f"{self.__class__.__name__}(namespaces={names})"
