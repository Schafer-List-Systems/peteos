from pathlib import Path
from typing import Dict, List, Optional, Tuple

from peteos.utils import get_logger
from .role import Role

_logger = get_logger(__name__)


class RoleManager:
    """Manages roles with registration and directory loading capabilities."""

    _roles: Dict[str, Role] = {}

    @classmethod
    def register_role(cls, role: Role) -> None:
        """
        Register a role manually.

        Args:
            role: The Role instance to register.
        """
        cls._roles[role.name] = role

    @classmethod
    def load_from_dir(cls, directory: str) -> List[str]:
        """
        Load all roles from a directory where subdirectories are role names.

        Each subdirectory should contain:
        - description.md (or config.json.description)
        - system_prompt.md (optional, or config.json.system_prompt)
        - config.json (optional, for required_tools, execution_environment, model)

        Args:
            directory: Path to directory containing role subdirectories.

        Returns:
            List of successfully loaded role names.
        """
        loaded_roles: List[str] = []
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        for sub_dir in dir_path.iterdir():
            if sub_dir.is_dir():
                try:
                    role = Role.load_from_path(str(sub_dir))
                    cls.register_role(role)
                    loaded_roles.append(role.name)
                except (FileNotFoundError, KeyError, ValueError) as e:
                    # Skip roles that can't be loaded
                    _logger.warning(f"Could not load role from {sub_dir}: {e}")

        return loaded_roles

    @classmethod
    def get_role(cls, name: str) -> Optional[Role]:
        """
        Get a role by name.

        Args:
            name: The role name to look up.

        Returns:
            The Role instance if found, None otherwise.
        """
        return cls._roles.get(name)

    @classmethod
    def list_roles(cls) -> List[Tuple[str, Role]]:
        """
        List all registered roles.

        Returns:
            List of (name, Role) tuples.
        """
        return list(cls._roles.items())
