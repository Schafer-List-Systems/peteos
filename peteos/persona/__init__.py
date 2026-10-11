"""Persona — the behavioural layer: roles, tools, and agents.

A role defines *who* is acting — its identity, capabilities (tools),
behaviour policy, and system prompt.  An agent binds a role to a tool
manager and orchestrates sessions on its behalf.

Classes: ``Agent``, ``Role``, ``RoleManager``, ``Tool``, ``ToolManager``.
"""

from peteos.config import ConfigManager
from peteos.utils import get_logger

_logger = get_logger(__name__)

# Set ROLES_DIR once on first import — used for role discovery
ConfigManager.set_variable("ROLES_DIR", "{CONFIG_DIR}/roles")

# Bootstrap roles from ROLES_DIR into RoleManager
from peteos.persona.rolemanager import RoleManager
roles_dir = ConfigManager.resolve_path("{ROLES_DIR}")
try:
    loaded_roles = RoleManager.load_from_dir(roles_dir)
    _logger.info("Loaded %d role(s) from %s", len(loaded_roles), roles_dir)
except FileNotFoundError:
    # roles directory doesn't exist — that's fine
    pass