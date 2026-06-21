from __future__ import annotations

import hashlib
import json
import re

from peteos.conversation.message import Message
from peteos.utils import get_logger

_logger = get_logger(__name__)


class ToolDefinitionsMessage(Message):
    """Tool definitions message built from registered hooks.

    Expects exactly two hooks:
    - ``TOOL_LIST_HOOK_NAME`` (mandatory): returns a JSON array of tool
      definitions, each with ``name``, ``description``, and ``parameters``.
    - ``TOOL_FILTER_HOOK_NAME`` (optional): returns a JSON array of tool
      name patterns to exclude from the tool list.

    This makes the message independent of any ToolManager reference at
    deserialization time — it only needs the hooks to be re-registered
    on the session.
    """

    TOOL_LIST_HOOK_NAME: str = "tool_list"
    TOOL_FILTER_HOOK_NAME: str = "tool_filter"

    def __init__(self, json_dict: dict | None = None) -> None:
        if json_dict is None:
            json_dict = {}
        json_dict.setdefault("role", "tool")
        super().__init__(json_dict)

    def materialize(
        self,
        materialized_hooks: dict[str, str] | None = None,
        content_map: dict[str, str] | None = None,
    ) -> None:
        """Materialize tool definitions from hooks.

        Args:
            materialized_hooks: Hook ID → content hash.
            content_map: Content hash → string text.
        """
        if materialized_hooks is None or content_map is None:
            return

        filter_patterns = self._materialize_tool_filter(materialized_hooks, content_map)
        tool_list = self._materialize_tool_list(materialized_hooks, content_map)

        filtered = self._filter_tool_list(tool_list, filter_patterns)
        self._json_dict["content"] = self._format_tool_list(filtered)

    def _materialize_tool_list(
        self,
        materialized_hooks: dict[str, str],
        content_map: dict[str, str],
    ) -> list[dict]:
        """Resolve the tool list hook and return the parsed tool definitions."""
        hook_id = self._tool_list_hook_id
        content_hash = materialized_hooks.get(hook_id)
        if content_hash is None:
            _logger.error(
                "ToolDefinitionsMessage: hook ID %r not found in materialized_hooks",
                hook_id,
            )
            return []
        text = content_map.get(content_hash)
        if text is None:
            _logger.error(
                "ToolDefinitionsMessage: content hash %r not found in content_map",
                content_hash,
            )
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            _logger.error(
                "ToolDefinitionsMessage: invalid JSON in tool list hook: %s", e
            )
            return []

    def _materialize_tool_filter(
        self,
        materialized_hooks: dict[str, str],
        content_map: dict[str, str],
    ) -> set[str]:
        """Resolve the tool filter hook and return excluded tool name patterns."""
        if self._tool_filter_hook_id not in materialized_hooks:
            return set()
        content_hash = materialized_hooks[self._tool_filter_hook_id]
        filter_text = content_map.get(content_hash)
        if not filter_text:
            return set()
        try:
            return set(json.loads(filter_text))
        except json.JSONDecodeError:
            _logger.error(
                "ToolDefinitionsMessage: invalid JSON in tool filter hook"
            )
            return set()

    @staticmethod
    def _make_hook_id(name: str) -> str:
        """Derive a deterministic hook ID from the name."""
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    @staticmethod
    def _filter_tool_list(
        tool_list: list[dict],
        filter_patterns: set[str],
    ) -> list[dict]:
        """Filter the tool list, excluding tools matching any regex pattern.

        Uses ``re.fullmatch`` — patterns like ``"internal_.*"`` will
        exclude tools whose names start with ``internal_``.

        Args:
            tool_list: The full list of tool definition dicts.
            filter_patterns: Regex patterns to exclude tools by name.

        Returns:
            The filtered tool list.
        """
        if not filter_patterns:
            return tool_list
        return [
            tool for tool in tool_list
            if not any(re.fullmatch(pattern, tool.get("name", "")) for pattern in filter_patterns)
        ]

    @staticmethod
    def _format_tool_list(tool_list: list[dict]) -> list[dict]:
        """Normalize the tool list to the expected content format.

        Ensures each tool dict has ``type="tool"`` with ``name``,
        ``description``, and ``parameters`` keys.

        Args:
            tool_list: The filtered list of tool definition dicts.

        Returns:
            The formatted tool list.
        """
        result = []
        for tool in tool_list:
            formatted: dict = {
                "type": "tool",
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            result.append(formatted)
        return result

    @property
    def _tool_filter_hook_id(self) -> str:
        """Pre-computed hook ID for the tool filter hook."""
        return self._make_hook_id(self.TOOL_FILTER_HOOK_NAME)

    @property
    def _tool_list_hook_id(self) -> str:
        """Pre-computed hook ID for the tool list hook."""
        return self._make_hook_id(self.TOOL_LIST_HOOK_NAME)
