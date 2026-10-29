"""Peteos configuration loader.

Resolves peteos.json from standard locations and bootstraps all
subsystems that need configuration (chatbot backends, roles, etc.).

Path resolution uses a variable expander — callers pass templates
with ``{VAR}`` tokens that are expanded against a shared dictionary.
"""

import os
from pathlib import Path
from typing import Dict, Optional

from peteos.utils import get_logger
from peteos.utils import json

_logger = get_logger(__name__)


def _resolve_config_path() -> tuple[Optional[str], Optional[str]]:
    """Resolve the peteos.json config file and its dedicated config directory.

    Discovery order (first found wins):
        1. PETEOS_CONFIG environment variable (full path)
        2. ./peteos.json (current working directory — project-local)
        3. $XDG_CONFIG_HOME/peteos/peteos.json (defaults to ~/.config/peteos/ — user-specific)
        4. /etc/peteos/peteos.json (system-wide — last resort)

    Returns:
        Tuple of (config_file, config_dir).
        config_file is the absolute path to peteos.json, or None if not found.
        config_dir is the dedicated config directory for this installation:
          - For dedicated locations (XDG, /etc): the parent directory of the config file.
          - For non-dedicated locations (env, cwd): config_file's parent / ".peteos".
    """
    # 1. Explicit env var
    env_path = os.environ.get("PETEOS_CONFIG")
    if env_path and Path(env_path).is_file():
        return (os.path.abspath(env_path), str(Path(env_path).parent / ".peteos"))

    # 2. Current working directory
    local_path = Path.cwd() / "peteos.json"
    if local_path.is_file():
        return (str(local_path), str(local_path.parent / ".peteos"))

    # 3. User-specific XDG config directory
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if not xdg_config:
        xdg_config = str(Path.home() / ".config")
    xdg_path = Path(xdg_config) / "peteos" / "peteos.json"
    if xdg_path.is_file():
        return (str(xdg_path), str(xdg_path.parent))

    # 4. System-wide config (last resort)
    system_path = "/etc/peteos/peteos.json"
    if Path(system_path).is_file():
        return (system_path, str(Path(system_path).parent))

    return (None, None)


def _load_config_file(filepath: str) -> dict:
    """Load and parse a peteos.json config file.

    Args:
        filepath: Absolute path to the JSON config file.

    Returns:
        Parsed JSON dict.
    """
    with open(filepath, "r") as f:
        return json.load(f)


class _CyclingDict(dict):
    """dict wrapper that expands values through format_map with cycle detection.

    When a key is retrieved via __getitem__, the value is passed through
    format_map using a fresh _CyclingDict. This means that if the value
    itself contains {tokens}, they are expanded recursively — with each
    branch of expansion getting its own copy of the seen set, so sibling
    expansions never interfere.

    A cycle (A→{B}, B→{A}) raises RecursionError on the second visit.
    """

    def __init__(self, data: dict, seen: frozenset):
        super().__init__(data)
        self._seen = seen

    def __getitem__(self, key: str) -> str:
        if key in self._seen:
            raise RecursionError(f"variable cycle detected: '{key}'")
        raw = super().__getitem__(key)
        return raw.format_map(_CyclingDict(self, self._seen | {key}))


class ConfigManager:
    """Singleton that manages configuration variables and path resolution.

    Subclasses should not inherit from this — use the class directly.

    Class variables hold the resolved config state after init().

    Variables are bootstrapped from environment variables prefixed with ``PETEOS_``.
    The prefix is stripped, so ``PETEOS_CONFIG_DIR`` becomes key ``CONFIG_DIR``.
    """

    _variables: Dict[str, str] = {}
    _config_dir: Optional[str] = None
    _config_file: Optional[str] = None

    @classmethod
    def set_variable(cls, key: str, value: str, overwrite: bool = False) -> bool:
        """Add a variable to the shared variable dictionary.

        When overwrite is False, the operation is a no-op if the key already
        exists and this method returns False. When overwrite is True, the
        value is always written and this method always returns True.

        Args:
            key: Variable name, used as ``{key}`` in templates.
            value: Expansion value for this variable.
            overwrite: If True, always write the value and return True.

        Returns:
            True if the value was written, False if it was not written
            because the key already exists and overwrite is False.
        """
        if key in cls._variables and not overwrite:
            return False
        cls._variables[key] = value
        return True

    @classmethod
    def get_variable(cls, key: str) -> str:
        """Retrieve a variable from the shared dictionary.

        Args:
            key: Variable name.

        Returns:
            The expansion value for this variable.

        Raises:
            KeyError: If the variable is not set.
        """
        return cls._variables[key]

    @classmethod
    def resolve_path(
        cls,
        template: str,
        overrides: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Resolve a path template to a filesystem path by expanding variables.

        The template is expanded using str.format_map with a cycling-aware
        mapping. ``{{`` and ``}}`` escape to literal braces. Missing variables
        raise KeyError. Overrides take precedence over the shared dictionary.

        Args:
            template: Path string containing ``{var}`` tokens to expand.
            overrides: Optional per-call variables that take precedence.

        Returns:
            Resolved Path object.

        Raises:
            KeyError: If a variable token is not found.
            RecursionError: If a cycle is detected during expansion.
        """
        working: Dict[str, str] = dict(cls._variables)
        if overrides is not None:
            working.update(overrides)

        mapping = _CyclingDict(working, frozenset())
        resolved = template.format_map(mapping)
        return Path(resolved)

    @classmethod
    def init(cls) -> bool:
        """Resolve config and bootstrap all subsystems.

        Discovers peteos.json, loads backends into ChatBotManager,
        and loads roles into RoleManager from the config directory's
        ``roles/`` subdirectory (if it exists).

        Returns:
            True if this call performed the initialization.
            False if already initialized (no-op).
        Raises:
            FileNotFoundError: If a config file was requested but not found.
            RuntimeError: If backend loading or detection fails.
        """
        # Guard against repeated initialization — singleton semantics
        if "CONFIG" in cls._variables:
            _logger.debug("ConfigManager.init() skipped — already initialized")
            return False

        # Allow PETEOS_DEBUG=1 env var to force verbose logging for diagnostics
        if os.environ.get("PETEOS_DEBUG") == "1":
            from peteos.utils.logger import setup_logging
            setup_logging(level="DEBUG", debug=True)

        # Bootstrap variables from PETEOS_* environment variables — strip the prefix
        for key, value in os.environ.items():
            if key.startswith("PETEOS_"):
                var_name = key[len("PETEOS_") :]
                cls._variables[var_name] = value
        if cls._variables:
            _logger.debug("[config] bootstrapped %d env variable(s)", len(cls._variables))

        # Discover peteos.json location — (config_file, config_dir) or (None, None) if absent
        config_file, config_dir = _resolve_config_path()
        if config_file is None:
            _logger.warning("No peteos.json config file found in standard locations")
            return True
        _logger.debug("Using peteos.json config from: %s", config_file)

        # Set CONFIG and CONFIG_DIR variables — used by resolve_path in templates
        cls.set_variable("CONFIG", config_file, overwrite=False)
        cls.set_variable("CONFIG_DIR", config_dir, overwrite=False)

        # Initialization complete
        return True


# Auto-initialize on import
ConfigManager.init()
