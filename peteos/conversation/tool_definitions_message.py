from __future__ import annotations

import json

from peteos.conversation.message import Message
from peteos.logger import get_logger

_logger = get_logger(__name__)


class ToolDefinitionsMessage(Message):
    """Tool definitions message built from registered hooks.

    A hook is registered on the session that returns a JSON string
    representing a list of serialized tool definitions. Each item in
    the list has the keys ``type``, ``name``, ``description``, and
    ``parameters``.

    During materialization, the hook outputs are resolved and stored
    directly as the message content.

    This makes the message independent of any ToolManager reference at
    deserialization time — it only needs the hook to be re-registered
    on the session.
    """

    def __init__(self, json_dict: dict) -> None:
        json_dict.setdefault("role", "tool")
        super().__init__(json_dict)

    def materialize(
        self,
        materialized_hooks: dict[str, str] | None = None,
        content_map: dict[str, str] | None = None,
    ) -> None:
        """Materialize tool definitions from hook outputs.

        Each hook returns a JSON string of serialized tool definitions
        (list of tool dicts with type, name, description, parameters).

        Args:
            materialized_hooks: Hook ID → content hash.
            content_map: Content hash → string text.
        """
        if materialized_hooks is None or content_map is None:
            return

        texts: list[str] = []
        for hook_id in self.hook_ids:
            content_hash = materialized_hooks.get(hook_id)
            if content_hash is None:
                _logger.error(
                    "ToolDefinitionsMessage: hook ID %r not found in materialized_hooks",
                    hook_id,
                )
                continue
            text = content_map.get(content_hash)
            if text is None:
                _logger.error(
                    "ToolDefinitionsMessage: content hash %r not found in content_map",
                    content_hash,
                )
                continue
            texts.append(text)

        self._json_dict["content"] = json.loads("\n".join(texts))