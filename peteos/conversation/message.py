import uuid
from datetime import datetime

from .message_registry import MessageRegistry
from peteos.utils import json


class ContentPart:
    """Wrapper around a single serialized content part dict.

    Follows a unified schema: exactly one data key is set per part,
    never more than one. All others are absent or null.

    Schema:
        {"type": "text", "text": str}
        {"type": "image", "source": {"type": "base64"|"url", "data": str, "media_type": str}}
        {"type": "video", "source": {"type": "base64"|"url", "data": str, "media_type": str}}
        {"type": "audio", "source": {"type": "base64"|"url", "data": str, "media_type": str}}
        {"type": "thinking", "text": str}
        {"type": "tool_use", "call_id": str, "name": str, "arguments": str}
        {"type": "tool_result", "call_id": str, "content": str}
        {"type": "tool", "name": str, "description": str, "parameters": dict}

    Example:
        >>> part = ContentPart({"type": "text", "text": "Hello"})
        >>> part.type
        'text'
        >>> part.text
        'Hello'
    """

    def __init__(self, json_dict: dict) -> None:
        """
        ContentPart objects shall only be created using the factory functions!

        Args:
            json_dict: The raw serialized content part dict.
        """
        self._json_dict = json_dict

    @staticmethod
    def create_text(text: str) -> "ContentPart":
        """Create a text content part."""
        return ContentPart({"type": "text", "text": text})

    @staticmethod
    def create_thinking(text: str) -> "ContentPart":
        """Create a thinking content part."""
        return ContentPart({"type": "thinking", "text": text})

    @staticmethod
    def create_image(source: dict) -> "ContentPart":
        """Create an image content part.

        Args:
            source: A source dict like {"type": "base64", "data": "...", "media_type": "image/png"}
                    or {"type": "url", "url": "https://..."}.
        """
        return ContentPart({"type": "image", "source": source})

    @staticmethod
    def create_video(source: dict) -> "ContentPart":
        """Create a video content part.

        Args:
            source: A source dict like {"type": "base64", "data": "...", "media_type": "video/mp4"}
                    or {"type": "url", "url": "https://..."}.
        """
        return ContentPart({"type": "video", "source": source})

    @staticmethod
    def create_audio(source: dict) -> "ContentPart":
        """Create an audio content part.

        Args:
            source: A source dict like {"type": "base64", "data": "...", "media_type": "audio/mpeg"}
                    or {"type": "url", "url": "https://..."}.
        """
        return ContentPart({"type": "audio", "source": source})

    @staticmethod
    def create_tool_use(call_id: str, name: str, arguments: str) -> "ContentPart":
        """Create a tool use content part.

        Args:
            call_id: The tool call ID.
            name: The tool name.
            arguments: JSON string of tool arguments.
        """
        return ContentPart({"type": "tool_use", "call_id": call_id, "name": name, "arguments": arguments})

    @staticmethod
    def create_tool_result(call_id: str, content: str) -> "ContentPart":
        """Create a tool result content part.

        Args:
            call_id: The tool call ID this result belongs to.
            content: The tool output text.
        """
        return ContentPart({"type": "tool_result", "call_id": call_id, "content": content})

    @staticmethod
    def create_tool(name: str, description: str, parameters: dict) -> "ContentPart":
        """Create a tool definition content part.

        Args:
            name: The tool name.
            description: The tool description.
            parameters: JSON schema dict for tool parameters.
        """
        return ContentPart({"type": "tool", "name": name, "description": description, "parameters": parameters})

    @staticmethod
    def create_pdf(source: dict) -> "ContentPart":
        """Create a PDF content part.

        Args:
            source: A source dict like {"type": "base64", "data": "...", "media_type": "application/pdf"}
                    or {"type": "url", "url": "https://..."}.
        """
        return ContentPart({"type": "pdf", "source": source})

    @property
    def raw_dict(self) -> dict:
        """Return the wrapped serialized dict."""
        return self._json_dict

    @property
    def type(self) -> str:
        """Return the content part type."""
        return self._json_dict["type"]

    @property
    def text(self) -> str | None:
        """Return the text content (for text/thinking parts)."""
        return self._json_dict.get("text")

    @property
    def source(self) -> dict | None:
        """Return the source dict (for image/video/audio parts)."""
        return self._json_dict.get("source")

    @property
    def call_id(self) -> str | None:
        """Return the tool call ID (for tool_use/tool_result parts)."""
        return self._json_dict.get("call_id")

    @property
    def name(self) -> str | None:
        """Return the tool name (for tool_use parts)."""
        return self._json_dict.get("name")

    @property
    def arguments(self) -> str | None:
        """Return the JSON arguments string (for tool_use parts)."""
        return self._json_dict.get("arguments")

    @property
    def content(self) -> str | None:
        """Return the result content string (for tool_result parts)."""
        return self._json_dict.get("content")

    def set_tool_result_content(self, content: str) -> None:
        """Set the result content for a tool_result part."""
        self._json_dict["content"] = content

    @property
    def description(self) -> str | None:
        """Return the tool description (for tool definition parts)."""
        return self._json_dict.get("description")

    @property
    def parameters(self) -> dict | None:
        """Return the tool parameters schema (for tool definition parts)."""
        return self._json_dict.get("parameters")


class Message:
    """Wrapper around a single serialized message dict.

    Follows the agentic_process pattern: the dict is the source of truth,
    this class is just a convenient wrapper.

    All subclasses are auto-registered in MessageRegistry.

    A message is dynamic when it has hook IDs. During materialization,
    dynamic messages are resolved via the content map.

    Example:
        >>> msg = Message({"role": "user", "content": [{"type": "text", "text": "Hi"}]})
        >>> msg.raw_dict
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}
    """

    def __init__(self, json_dict: dict) -> None:
        """
        Message objects shall only be created using the factory functions!

        Args:
            json_dict: The raw serialized message dict.
        """
        if "_type" not in json_dict:
            json_dict["_type"] = self.__class__.__name__
        if "id" not in json_dict:
            json_dict["id"] = str(uuid.uuid4())
        json_dict.setdefault("_hook_ids", [])
        if "creation_timestamp" not in json_dict:
            json_dict["creation_timestamp"] = datetime.now().isoformat()
        self._json_dict = json_dict

    @staticmethod
    def create(role: str, content_parts: list[ContentPart], metadata: dict | None = None) -> "Message":
        """Create a message with the given role, content parts, and optional metadata."""
        json_dict: dict = {"role": role, "content": [part.raw_dict for part in content_parts]}
        if metadata is not None:
            json_dict["metadata"] = metadata
        return Message(json_dict)

    @staticmethod
    def from_dict(json_dict: dict, strict: bool = True) -> "Message":
        """Reconstruct the correct message subclass via the registry.

        Uses ``_type`` from the dict to look up the class. When
        ``strict=True`` (default), raises ``ValueError`` if no
        registered class matches. When ``strict=False``, falls back to
        the base ``Message`` class.

        Args:
            json_dict: The raw serialized message dict.
            strict: If True, raise on missing class; if False, fall back.

        Returns:
            An instance of the appropriate Message subclass.

        Raises:
            ValueError: If ``strict=True`` and the class is not registered.
        """
        msg_type = json_dict.get("_type", "Message")
        cls = MessageRegistry.get(msg_type)
        if cls is None and strict:
            raise ValueError(
                f"Registered message class not found: {msg_type!r}"
            )
        return (cls or Message)(json_dict)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        MessageRegistry.register(cls)

    @property
    def raw_dict(self) -> dict:
        """Return the wrapped serialized dict."""
        return self._json_dict

    @property
    def role(self) -> str:
        """Return the message role."""
        return self._json_dict.get("role", "")

    @property
    def id(self) -> str:
        """Return the message ID."""
        return self._json_dict.get("id", "")

    @property
    def hook_ids(self) -> list[str]:
        """Return the list of hook IDs associated with this message.

        If non-empty, this is a dynamic message.
        """
        return self._json_dict.get("_hook_ids", [])

    def materialize(self, materialized_hooks: dict[str, str] | None = None, content_map: dict[str, str] | None = None) -> None:
        """Materialize this message using the content map and materialized hooks.

        For dynamic messages, resolves hook IDs to content hashes
        stored in the content map. For static messages, returns a copy.

        Args:
            materialized_hooks: Hook ID → content hash map (optional).
            content_map: The shared content hash → string map (optional).
        """
        # TODO: implement dynamic resolution via hook IDs and content_map

    @property
    def creation_timestamp(self) -> datetime:
        """Return the message creation timestamp."""
        return datetime.fromisoformat(self._json_dict["creation_timestamp"])

    @property
    def metadata(self) -> dict:
        """Return the message metadata dict."""
        return self._json_dict.get("metadata", {})

    @property
    def content(self) -> list[ContentPart]:
        """Return the list of content parts for this message."""
        return [ContentPart(part) for part in self._json_dict.get("content", [])]

    def printable(self) -> str:
        """Return a human-readable string of the message content.

        Text/thinking content is shown as-is. Media parts show type and media_type.
        Tool use and tool result parts show the tool name and content length.
        Multi-part messages use newlines between parts.
        """
        parts: list[str] = []
        for part in self.content:
            t = part.type
            if t in ("text", "thinking"):
                parts.append(part.text or "")
            elif t in ("image", "video", "audio"):
                parts.append(f"[{t.capitalize()}: media_type={part.source.get('media_type', 'unknown')}]" if part.source else f"[{t.capitalize()}]")
            elif t == "tool_use":
                parts.append(f"[ToolCall: {part.name} id={part.call_id}]")
            elif t == "tool_result":
                content = part.content or ""
                parts.append(f"[ToolResult of {part.call_id or 'unknown'}: {len(content)} chars]")
            elif t == "tool":
                params = part.parameters or {}
                parts.append(f"[ToolDef: {part.name} params={len(params)} fields]")
            else:
                parts.append(f"[Unknown: {t}]")
        return "\n".join(parts)

    def _compute_token_count(self, encoding: str = "cl100k_base") -> int:
        """Compute token count from serialized content without caching.

        Args:
            encoding: Tiktoken encoding to use.

        Returns:
            Token count as an integer.
        """
        from peteos.utils.tiktoken import count_tiktoken

        serialized = self._json_dict.get("content", [])
        full_text = f"{self._json_dict.get('role', '')}: {json.dumps(serialized, ensure_ascii=False) if serialized else ''}"
        return count_tiktoken(full_text, encoding)

    def count_tokens(self, encoding: str = "cl100k_base") -> int:
        """Count tokens in this message's content.

        Uses cached token count from metadata if available, otherwise
        computes and caches the result.

        Args:
            encoding: Tiktoken encoding to use.

        Returns:
            Token count as an integer.
        """
        if "token_count" in self._json_dict:
            return self._json_dict["token_count"]
        token_count = self._compute_token_count(encoding)
        self._json_dict["token_count"] = token_count
        return token_count


# Register the base Message class itself (only subclasses get auto-registered via __init_subclass__)
MessageRegistry.register(Message)
