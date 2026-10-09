"""Peteos configuration loader.

Resolves peteos.json from standard locations and bootstraps all
subsystems that need configuration (chatbot backends, roles, etc.).
"""

import os
from pathlib import Path
from typing import Optional

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


class ConfigManager:
    """Singleton that resolves the Peteos configuration directory and bootstraps subsystems.

    Class variables hold the resolved config state after init().
    """

    _config_dir: Optional[str] = None
    _config_file: Optional[str] = None

    @classmethod
    async def init(cls) -> bool:
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
        if cls._config_file is not None:
            _logger.debug("ConfigManager.init() skipped — already initialized")
            return False

        # Allow PETEOS_DEBUG=1 env var to force verbose logging for diagnostics
        if os.environ.get("PETEOS_DEBUG") == "1":
            from peteos.utils.logger import setup_logging
            setup_logging(level="DEBUG", debug=True)

        # Discover peteos.json location — (config_file, config_dir) or (None, None) if absent
        config_file, config_dir = _resolve_config_path()
        if config_file is None:
            _logger.warning("No peteos.json config file found in standard locations")
            return True

        _logger.debug("Using peteos.json config from: %s", config_file)

        # Persist resolved config file and its dedicated directory to the singleton class
        cls._config_file = config_file
        cls._config_dir = config_dir

        # Load the JSON file
        json_obj = _load_config_file(config_file)

        # Bootstrap chatbot backends into ChatBotManager for invoke_agent() model selection
        from peteos.chatbot import ChatBotManager
        await ChatBotManager.load_from_json(json_obj)
        loaded_backends = list(ChatBotManager._backends.keys())
        _logger.info("Loaded %d backend(s) from peteos.json", len(loaded_backends))

        # Set Agent.agent_base — used by Agent.__init__ to derive the session directory
        agent_base = os.path.join(cls._config_dir, "agents")
        from peteos.persona.agent import Agent
        Agent.agent_base = agent_base

        # Bootstrap roles from {config_dir}/roles/ into RoleManager
        roles_dir = os.path.join(cls._config_dir, "roles")
        try:
            from peteos.persona.rolemanager import RoleManager
            loaded_roles = RoleManager.load_from_dir(roles_dir)
            _logger.info("Loaded %d role(s) from %s", len(loaded_roles), roles_dir)
        except FileNotFoundError:
            # roles directory doesn't exist — that's fine
            pass

        # Initialization complete — callers can query ConfigManager._config_file
        return True
