import json
from pathlib import Path
from typing import Optional


class Role:
    """A role with name, description, and optional system prompt."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Role.

        Args:
            name: The name of the role.
            description: Description of the role.
            system_prompt: Optional system prompt.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt

    @staticmethod
    def load_from_dict(data: dict) -> "Role":
        """
        Creates a new role from a dictionary.

        Args:
            data: Dictionary with 'name', 'description', and optional 'system_prompt' keys.

        Returns:
            A new Role instance.
        """
        name = data["name"]
        description = data["description"]
        system_prompt = data.get("system_prompt")
        return Role(name=name, description=description, system_prompt=system_prompt)

    @staticmethod
    def load_from_path(path: str) -> "Role":
        """
        Creates a new role from a directory.

        The name is taken from the path suffix. Description is in 'description.md',
        system prompt (if present) is in 'system_prompt.md'.

        Args:
            path: Directory path containing the role files.

        Returns:
            A new Role instance.
        """
        path_obj = Path(path)
        name = path_obj.suffix.lstrip(".") if path_obj.suffix else path_obj.name

        description_path = path_obj / "description.md"
        description = description_path.read_text() if description_path.exists() else ""

        system_prompt_path = path_obj / "system_prompt.md"
        system_prompt = system_prompt_path.read_text().strip() if system_prompt_path.exists() else None

        return Role(name=name, description=description, system_prompt=system_prompt)
