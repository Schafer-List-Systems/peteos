import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional


class SessionState:
    """Mutable key-value store for leaving intermediate state.

    Variables store strings. Backed by a dictionary inside the session's
    JSON dict so serializing the session also serializes its state.

    Five distinct methods with clear semantics:

    - get(name): returns the value or None if not found
    - create(name, value): creates a new variable (raises if exists)
    - update(name, old_value, new_value): compare-and-swap (both non-None)
    - delete(name): removes a variable (raises if not found)
    - list(): returns all variable names
    """

    def __init__(self, data: dict[str, str] | None = None) -> None:
        self._data: dict[str, str] = data if data is not None else {}

    def get(self, name: str) -> Optional[str]:
        """Get a variable's value."""
        return self._data.get(name)

    def create(self, name: str, value: str) -> None:
        """Create a new variable."""
        if name in self._data:
            raise ValueError(f"Variable '{name}' already exists")
        if value is None:
            raise ValueError("Value must not be None")
        self._data[name] = value

    def update(self, name: str, old_value: str, new_value: str) -> None:
        """Update with compare-and-swap semantics."""
        if old_value is None:
            raise ValueError("old_value must not be None")
        if new_value is None:
            raise ValueError("new_value must not be None")
        if name not in self._data:
            raise KeyError(f"Variable '{name}' does not exist")
        current = self._data[name]
        if current != old_value:
            raise ValueError(
                f"Variable '{name}' has value {current!r}, expected {old_value!r}"
            )
        self._data[name] = new_value

    def delete(self, name: str) -> None:
        """Delete a variable."""
        if name not in self._data:
            raise KeyError(f"Variable '{name}' does not exist")
        del self._data[name]

    def list(self) -> list[str]:
        """Return a list of all variable names."""
        return list(self._data.keys())

    def to_dict(self) -> dict[str, str]:
        """Return the underlying data dict for serialization."""
        return self._data


from peteos.conversation.context import Context
from peteos.conversation.message import Message

from .system_prompt_message import SystemPromptMessage
from .tool_definitions_message import ToolDefinitionsMessage

from peteos.utils import get_logger

_logger = get_logger(__name__)


class Session:
    """A session representing a single interaction thread.

    Wraps a JSON dict as the source of truth.
    """

    def __init__(self, parent_dir: str, json_dict: dict | None = None) -> None:
        """
        Args:
            parent_dir: The agent directory (parent of the session subdirectory).
            json_dict: The raw serialized session dict.
        """
        if json_dict is None:
            json_dict = {}

        self._parent_dir = parent_dir
        self._json_dict = json_dict
        self._json_dict.setdefault(
            "uuid",
            f"{datetime.date.today().strftime('%Y%m%d')}-{uuid.uuid4()}",
        )
        self._json_dict.setdefault("active_context_id", None)
        self._json_dict.setdefault("auto_approve_tools", [])
        self._json_dict.setdefault("state_data", {})
        self._autosave = True
        self._hooks: dict[str, Callable[[], str]] = {}
        self._active_context: Context | None = None
        self._state: SessionState = SessionState(self._json_dict.setdefault("state_data", {}))

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
        """Materialize all hooks and return a map from hook ID to content hash.

        Side effect: populates the active context's content_map with hash →
        content_text so messages can resolve hook outputs during materialize().
        """
        hook_to_hash: dict[str, str] = {}
        for hook_id, callback in self._hooks.items():
            content = callback()
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            hook_to_hash[hook_id] = content_hash
            active = self._active_context
            if active:
                active.content_map[content_hash] = content
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

    @property
    def state(self) -> SessionState:
        """Return the session's mutable key-value store.

        Lives inside the session's JSON dict as ``state_data`` so
        serializing the session automatically serializes its state.
        """
        return self._state

    def set_active_context(self, context: Context) -> None:
        """Set the active context, synchronizing both the JSON dict ID and the Python object reference."""
        self.save_active_context()
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

    def rolling_sequence_window(self, count: int) -> Context | None:
        """Apply a rolling sequence window to the active context.

        Delegates to ``Context.rolling_sequence_window(count)`` and sets
        the result as the new active context only if messages were actually
        dropped.

        Args:
            count: Number of non-special messages to keep from the end.

        Returns:
            The new active Context, or None if the context was already
            within the window or there is no active context.
        """
        ctx = self.active_context
        if ctx is None:
            return None
        new_ctx = ctx.rolling_sequence_window(count)
        if new_ctx is not None:
            self.set_active_context(new_ctx)
        return new_ctx

    def rolling_token_window(
        self,
        max_tokens: int,
        encoding: str = "cl100k_base",
    ) -> Context | None:
        """Apply a rolling token window to the active context.

        Delegates to ``Context.rolling_token_window(max_tokens)`` and sets
        the result as the new active context only if messages were actually
        dropped.

        Args:
            max_tokens: Maximum total token count for the resulting context.
            encoding: Tiktoken encoding name.

        Returns:
            The new active Context, or None if the context was already
            within the window or there is no active context.
        """
        ctx = self.active_context
        if ctx is None:
            return None
        new_ctx = ctx.rolling_token_window(max_tokens, encoding)
        if new_ctx is not None:
            self.set_active_context(new_ctx)
        return new_ctx

    @property
    def auto_approve_tools(self) -> list[str]:
        """Return the list of tool names auto-approved by this session."""
        return self.raw_dict["auto_approve_tools"]

    @classmethod
    def create(
        cls,
        parent_dir: str,
        system_prompt_message: SystemPromptMessage | None = None,
        tool_definitions_message: ToolDefinitionsMessage | None = None,
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
        _logger.info("Created session %s in %s", session.uuid, session.session_dir)
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

    def save_active_context(self) -> None:
        """Save the active context to disk if autosave is enabled and the session directory exists."""
        if not self._autosave:
            return
        if self.session_dir.exists() and self._active_context is not None:
            self._active_context.save(self.session_dir)

    def save(self) -> None:
        """Save the session and its active context to disk."""
        session_path = self.session_dir / "session.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, "w") as f:
            json.dump(self._json_dict, f, indent=2)
        _logger.info("Saved session.json to %s", session_path)

        self.save_active_context()

    def __del__(self) -> None:
        """Auto-save the session and its active context on destruction."""
        try:
            self.save()
        except Exception:
            pass
