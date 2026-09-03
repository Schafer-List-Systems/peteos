"""LLM-based semantic string comparison for OAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.error import Error

if TYPE_CHECKING:
    pass


class AgenticStringComparator(AgenticObject):
    """You are a String Comparator answering questions with true or false."""

    _instance: AgenticStringComparator | None = None

    @classmethod
    def instance(cls) -> AgenticStringComparator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _question(self, prompt: str) -> bool:
        """Ask the judge a yes/no question."""
        result: bool | Error = await self.invoke_agent(
            prompt,
            output_schema=bool,
            persistent_thread_id=None,
        )
        if isinstance(result, Error):
            raise result.message
        return bool(result)

    async def contains(self, text: str, substring: str) -> bool:
        """Does the text contain this information? (LLM-evaluated semantic containment.)"""
        return await self._question(
            f"Does the following text contain the information {substring!r}?\n\nText: {text}"
        )

    async def is_same(self, a: str, b: str) -> bool:
        """Are these two texts semantically equivalent?"""
        return await self._question(
            "Are these two texts semantically equivalent (they express the same meaning)?\n\n"
            f"Text A: {a!r}\n\nText B: {b!r}"
        )

    async def contradicts(self, text: str, claim: str) -> bool:
        """Does the text contradict the claim?"""
        return await self._question(
            f"Does the following text contradict the claim {claim!r}?\n\nText: {text}"
        )

    async def is_complete(self, text: str, required_info: str) -> bool:
        """Does the text contain all required information?"""
        return await self._question(
            f"Does the following text contain the required information: {required_info!r}?\n\nText: {text}"
        )
