from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable


@dataclass
class RecursiveTokenCounter:
    """Accumulates token counts across nested agent invocations via invocation hooks.

    Create one with :meth:`make_counter`, then pass the returned hooks dict to
    :meth:`~peteos.oap.base.AgenticObject.invoke_agent`.  After the call chain
    returns, read ``total_delta`` for the total token footprint consumed by the
    entire invocation tree.

    Usage::

        counter, hooks = RecursiveTokenCounter.make_counter()
        result = await target.invoke_agent(prompt=prompt, hooks=hooks)
        print(f"Tokens used: {counter.total_delta}")

    """
    _before_tokens: list[int] = field(default_factory=list, repr=False)
    _after_tokens: list[int] = field(default_factory=list, repr=False)

    @property
    def total_delta(self) -> int:
        """Total token footprint across the entire invocation tree.

        Sum of all after-tokens minus sum of all before-tokens.  If more
        invocations were recorded in ``on_invoke`` than in ``on_invoke_complete``
        (e.g. because an invocation was prevented by an ``on_invoke`` hook
        returning a prevention string), the unpaired before values are silently
        ignored — the difference is capped at zero.
        """
        total_after = sum(self._after_tokens)
        total_before = sum(self._before_tokens)
        return max(0, total_after - total_before)

    @classmethod
    def make_counter(cls) -> tuple["RecursiveTokenCounter", dict[str, list[Callable]]]:
        """Create a counter and a pre-configured hooks dictionary.

        The returned hooks capture the counter in their closures, so every
        agent invocation — root, sub-agents, sub-sub-agents — fires both
        callbacks and accumulates into the same counter.

        Returns:
            A tuple of ``(counter, hooks)`` ready to pass to
            :meth:`~peteos.oap.base.AgenticObject.invoke_agent`.
        """
        counter = cls()

        hooks: dict[str, list[Callable[..., Any]]] = {
            "on_invoke": [
                lambda ctx, c=counter: c._before_tokens.append(
                    ctx["session"].active_context.total_token_count()
                ),
            ],
            "on_invoke_complete": [
                lambda ctx, c=counter: c._after_tokens.append(
                    ctx["session"].active_context.total_token_count()
                ),
            ],
        }
        return counter, hooks
