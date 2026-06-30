from __future__ import annotations

from .message import ContentPart, Message
from peteos.utils import get_logger

_logger = get_logger(__name__)


class SystemPromptMessage(Message):
    """System prompt message built from registered hooks.

    Hooks are registered via ``Session.register_hook()``. During
    materialization, all hook texts are assembled into a single
    text ContentPart as the message content.
    """

    def __init__(self, json_dict: dict) -> None:
        json_dict.setdefault("role", "system")
        super().__init__(json_dict)

    @staticmethod
    def create(static_text: str | None = None) -> "SystemPromptMessage":
        """Create a new system prompt message.

        Args:
            static_text: Optional base text. The content is populated
                immediately if no hooks will be registered later.

        Returns:
            A new SystemPromptMessage instance.
        """
        json_dict: dict = {}
        if static_text:
            json_dict["_static_text"] = static_text
        msg = SystemPromptMessage(json_dict)
        if not static_text:
            return msg
        msg._json_dict["content"] = [
            ContentPart.create_text(static_text).raw_dict
        ]
        return msg

    def materialize(
        self,
        materialized_hooks: dict[str, str] | None = None,
        content_map: dict[str, str] | None = None,
    ) -> None:
        """Materialize the system prompt from static text + hook outputs.

        Args:
            materialized_hooks: Hook ID → content hash.
            content_map: Content hash → string text.
        """
        if materialized_hooks is None or content_map is None:
            return

        texts: list[str] = []
        static = self._json_dict.get("_static_text")
        if static:
            texts.append(static)

        for hook_id in self.hook_ids:
            content_hash = materialized_hooks.get(hook_id)
            if content_hash is None:
                _logger.error(
                    "SystemPromptMessage: hook ID %r not found in materialized_hooks",
                    hook_id,
                )
                continue
            text = content_map.get(content_hash)
            if text is None:
                _logger.error(
                    "SystemPromptMessage: content hash %r not found in content_map",
                    content_hash,
                )
                continue
            texts.append(text)

        self._json_dict["content"] = [
            ContentPart.create_text("\n\n".join(texts)).raw_dict
        ]
