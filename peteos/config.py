"""Peteos configuration loader.

Resolves peteos.json from standard locations and bootstraps all
subsystems that need configuration (chatbot backends, roles, etc.).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from peteos.utils import get_logger

_logger = get_logger(__name__)


@dataclass
class ConfigResult:
    """Result of loading the Peteos configuration."""
    config_dir: Optional[str] = None
    agent_dir: Optional[str] = None
    loaded_backends: list = field(default_factory=list)
    loaded_roles: list = field(default_factory=list)


class ConfigManager:
    """Resolves the Peteos configuration directory and bootstraps subsystems.

    Discovery order (first found wins):
        1. PETEOS_CONFIG environment variable (full path)
        2. $XDG_CONFIG_HOME/peteos/peteos.json (defaults to ~/.config/peteos/)
        3. /etc/peteos/peteos.json
        4. ./peteos.json (current working directory)
    """

    @staticmethod
    def resolve_config_path() -> Optional[str]:
        """Resolve the path to peteos.json using standard discovery order.

        Returns:
            Absolute path to peteos.json, or None if not found.
        """
        # 1. Explicit env var
        env_path = os.environ.get("PETEOS_CONFIG")
        if env_path and Path(env_path).is_file():
            return os.path.abspath(env_path)

        # 2. XDG config directory
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if not xdg_config:
            xdg_config = str(Path.home() / ".config")
        xdg_path = Path(xdg_config) / "peteos" / "peteos.json"
        if xdg_path.is_file():
            return str(xdg_path)

        # 3. System config
        system_path = "/etc/peteos/peteos.json"
        if Path(system_path).is_file():
            return system_path

        # 4. Current working directory
        local_path = Path.cwd() / "peteos.json"
        if local_path.is_file():
            return str(local_path)

        return None

    @staticmethod
    def _load_config_file(filepath: str) -> dict:
        """Load and parse a peteos.json config file.

        Args:
            filepath: Absolute path to the JSON config file.

        Returns:
            Parsed JSON dict.
        """
        with open(filepath, "r") as f:
            return json.load(f)

    @classmethod
    async def init(cls) -> ConfigResult:
        """Resolve config and bootstrap all subsystems.

        Discovers peteos.json, loads backends into ChatBotManager,
        and loads roles into RoleManager from the config directory's
        ``roles/`` subdirectory (if it exists).

        Returns:
            ConfigResult with the config directory and loaded items.
        """
        filepath = cls.resolve_config_path()
        if filepath is None:
            _logger.warning("No peteos.json config file found in standard locations")
            return ConfigResult()

        config_dir = str(Path(filepath).parent)
        json_obj = cls._load_config_file(filepath)

        # Bootstrap chatbot backends
        from peteos.chatbot import ChatBotManager
        await ChatBotManager.load_from_json(json_obj)
        loaded_backends = list(ChatBotManager._backends.keys())
        _logger.info("Loaded %d backend(s) from peteos.json", len(loaded_backends))

        # Set Agent.agent_base before role loading (roles create Agent instances)
        agent_base = os.path.join(config_dir, "agents")
        from peteos.persona.agent import Agent
        Agent.agent_base = agent_base

        # Bootstrap roles from {config_dir}/roles/
        roles_dir = os.path.join(config_dir, "roles")
        loaded_roles: list = []
        try:
            from peteos.persona.rolemanager import RoleManager
            loaded_roles = RoleManager.load_from_dir(roles_dir)
            _logger.info("Loaded %d role(s) from %s", len(loaded_roles), roles_dir)
        except FileNotFoundError:
            # roles directory doesn't exist — that's fine
            pass
        except ImportError:
            _logger.debug("RoleManager not available, skipping role loading")

        return ConfigResult(
            config_dir=config_dir,
            agent_dir=agent_base,
            loaded_backends=loaded_backends,
            loaded_roles=loaded_roles,
        )
