import hashlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from peteos.conversation.context import Context
from peteos.conversation.message import Message

if TYPE_CHECKING:
    from peteos.conversation.system_prompt_message import SystemPromptMessage
    from peteos.conversation.tool_definitions_message import ToolDefinitionsMessage


class Session:
    """A session representing a single interaction thread.

    Wraps a JSON dict as the source of truth.
    """

    def __init__(self, parent_dir: str, json_dict: dict = {}) -> None:
        """
        Args:
            parent_dir: The agent directory (parent of the session subdirectory).
            json_dict: The raw serialized session dict.
        """
        self._parent_dir = parent_dir
        self._json_dict = json_dict
        self._json_dict.setdefault("uuid", str(uuid.uuid4()))
        self._json_dict.setdefault("active_context_id", None)
        self._json_dict.setdefault("auto_approve_tools", [])
        self._hooks: dict[str, Callable[[], str]] = {}
        self._active_context: Context | None = None

    @property
    def session_dir(self) -> Path:
        """Return the session directory path (parent_dir / uuid)."""
        return Path(self._parent_dir) / self.uuid

    def register_hook(self, message: Message, name: str, callback: Callable[[], str]) -> str:
        """Register a hook on a message in the active context."""
        hook_id = self._make_hook_id(name)
        self._hooks[hook_id] = callback
        message.raw_dict["_hook_ids"].append(hook_id)
        if self._active_context is not None:
            self._active_context._update_hook_index()
        return hook_id

    def _make_hook_id(self, name: str) -> str:
        """Derive a deterministic hook ID from the name."""
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def _materialize_hooks(self) -> dict[str, str]:
        """Materialize all hooks and return a map from hook ID to content hash."""
        hook_to_hash: dict[str, str] = {}
        for hook_id, callback in self._hooks.items():
            content = callback()
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            hook_to_hash[hook_id] = content_hash
        return hook_to_hash

    def materialize(self) -> None:
        """Materialize all messages in the active context.

        Runs all hooks, computes string return values, hashes them,
        and stores them in the active context's content map. Then
        materializes each dynamic message via its materialize() method.
        """
        active = self.active_context
        materialized_hooks = self._materialize_hooks()
        dynamic_messages = active._gather_dynamic_messages()
        for msg in dynamic_messages:
            msg.materialize(materialized_hooks, active.content_map)

    @property
    def raw_dict(self) -> dict:
        """Return the wrapped serialized dict."""
        return self._json_dict

    @property
    def uuid(self) -> str:
        """Return the session UUID."""
        return self.raw_dict["uuid"]

    def set_active_context(self, context: Context) -> None:
        """Set the active context, synchronizing both the JSON dict ID and the Python object reference."""
        self._active_context = context
        self.raw_dict["active_context_id"] = context.id

    @property
    def active_context(self) -> Context | None:
        """Return the currently active context Python object."""
        return self._active_context

    @property
    def active_context_id(self) -> str | None:
        """Return the active context ID from the JSON dict."""
        return self.raw_dict["active_context_id"]

    @property
    def auto_approve_tools(self) -> list[str]:
        """Return the list of tool names auto-approved by this session."""
        return self.raw_dict["auto_approve_tools"]

    @classmethod
    def create(
        cls,
        parent_dir: str,
        system_prompt_message: "SystemPromptMessage | None" = None,
        tool_definitions_message: "ToolDefinitionsMessage | None" = None,
    ) -> "Session":
        """Create a new session with a freshly created context.

        Args:
            parent_dir: The agent directory.
            system_prompt_message: Optional system prompt for this session.
            tool_definitions_message: Optional tool definitions for this session.

        Returns:
            A new Session instance with the active context set.
        """
        session = cls(parent_dir)
        session._active_context = Context.create(
            system_prompt_message=system_prompt_message,
            tool_definitions_message=tool_definitions_message,
        )
        session.raw_dict["active_context_id"] = session._active_context.id
        return session

    @classmethod
    def load(cls, parent_dir: str, session_uuid: str) -> "Session":
        """Load a session from a directory containing session.json.

        The session directory is ``parent_dir / session_uuid``.

        Args:
            parent_dir: The agent directory.
            session_uuid: The session's UUID (also the subdirectory name).

        Raises:
            FileNotFoundError: If session.json does not exist in the directory.
        """
        session_path = Path(parent_dir) / session_uuid / "session.json"
        if not session_path.exists():
            raise FileNotFoundError(f"session.json not found in {session_path}")

        with open(session_path, "r") as f:
            json_dict = json.load(f)

        session = cls(parent_dir, json_dict)
        if json_dict.get("active_context_id"):
            context_path = session.session_dir / (json_dict["active_context_id"] + ".json")
            session._active_context = Context.load(context_path)
        return session

    def save(self) -> None:
        """Save the session and its active context to disk."""
        session_path = self.session_dir / "session.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, "w") as f:
            json.dump(self._json_dict, f, indent=2)

        if self._active_context is not None:
            self._active_context.save(self.session_dir)
