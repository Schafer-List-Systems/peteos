import hashlib
import json
import uuid
from pathlib import Path
from typing import Callable

from peteos.conversation.context import Context
from peteos.conversation.message import Message


class Session:
    """A session representing a single interaction thread.

    Wraps a JSON dict as the source of truth.
    """

    def __init__(self, directory_path: str, json_dict: dict) -> None:
        """
        Args:
            directory_path: Path to the session directory on disk.
            json_dict: The raw serialized session dict.
        """
        self._directory_path = directory_path
        self._json_dict = json_dict
        self._json_dict.setdefault("uuid", str(uuid.uuid4()))
        self._json_dict.setdefault("active_context_id", None)
        self._hooks: dict[str, Callable[[], str]] = {}

        if self._json_dict.get("active_context_id"):
            context_path = Path(directory_path) / (self._json_dict["active_context_id"] + ".json")
            with open(context_path, "r") as f:
                self._active_context = Context.load_from_dict(json.load(f))
        else:
            self._active_context = Context.create()
            self._json_dict["active_context_id"] = self._active_context.id

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
        return self._json_dict["uuid"]

    def set_active_context(self, context: Context) -> None:
        """Set the active context, synchronizing both the JSON dict ID and the Python object reference."""
        self._active_context = context
        self._json_dict["active_context_id"] = context.id

    @property
    def active_context(self) -> Context | None:
        """Return the currently active context Python object."""
        return self._active_context

    @property
    def active_context_id(self) -> str | None:
        """Return the active context ID from the JSON dict."""
        return self._json_dict.get("active_context_id")

    @classmethod
    def create(cls, directory_path: str) -> "Session":
        """Create a new empty session.

        The session directory is created (or reused) but nothing is
        written to disk until ``save_to_file()`` is called.

        Args:
            directory_path: Path to the session directory on disk.

        Returns:
            A new empty Session instance.
        """
        return cls(directory_path, {})

    @classmethod
    def load_from_file(cls, directory_path: str) -> "Session":
        """Load a session from a directory containing session.json.

        Args:
            directory_path: Path to the session directory.

        Raises:
            FileNotFoundError: If session.json does not exist in the directory.
        """
        session_path = Path(directory_path) / "session.json"
        if not session_path.exists():
            raise FileNotFoundError(f"session.json not found in {directory_path}")

        with open(session_path, "r") as f:
            json_dict = json.load(f)

        return cls(directory_path, json_dict)

    def save_to_file(self) -> None:
        """Save the session and its active context to disk.

        Writes ``session.json`` and the active context file.
        """
        session_path = Path(self._directory_path) / "session.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, "w") as f:
            json.dump(self._json_dict, f, indent=2)

        if self._active_context is not None:
            self._active_context.save_to_file(self._directory_path)
