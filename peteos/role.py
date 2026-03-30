import json
from pathlib import Path
from typing import Optional


class Role:
    """A role with name, description, optional system prompt, and configuration."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: Optional[str] = None,
        required_tools: Optional[list[str]] = None,
        execution_environment: str = "REPL"
    ):
        """
        Initialize Role.

        Args:
            name: The name of the role.
            description: Description of the role.
            system_prompt: Optional system prompt.
            required_tools: Optional list of tool names required by this role.
            execution_environment: Name of the execution environment (default: "REPL").
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.required_tools = required_tools if required_tools is not None else []
        self.execution_environment = execution_environment

    @staticmethod
    def load_from_dict(data: dict) -> "Role":
        """
        Creates a new role from a dictionary.

        Args:
            data: Dictionary with 'name', 'description', and optional 'system_prompt',
                  'required_tools', 'execution_environment' keys.

        Returns:
            A new Role instance.
        """
        name = data["name"]
        description = data["description"]
        system_prompt = data.get("system_prompt")
        required_tools = data.get("required_tools", [])
        execution_environment = data.get("execution_environment", "REPL")
        return Role(
            name=name,
            description=description,
            system_prompt=system_prompt,
            required_tools=required_tools,
            execution_environment=execution_environment
        )

    @staticmethod
    def load_config_from_path(role_path: str) -> dict:
        """
        Load optional config.json from role directory.

        Args:
            role_path: Directory path containing the role files.

        Returns:
            Dictionary with config values, or empty dict if config.json doesn't exist.
        """
        config_path = Path(role_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def load_from_path(path: str) -> "Role":
        """
        Creates a new role from a directory.

        Description is loaded from 'description.md' (or 'config.json' as fallback).
        System prompt is loaded from 'system_prompt.md' (or 'config.json' as fallback).
        Required tools and execution environment come from 'config.json' only.

        Markdown files have precedence over config.json entries.

        Args:
            path: Directory path containing the role files.

        Returns:
            A new Role instance.
        """
        path_obj = Path(path)
        name = path_obj.suffix.lstrip(".") if path_obj.suffix else path_obj.name

        config = Role.load_config_from_path(path)

        # Markdown files have precedence over config.json
        description_path = path_obj / "description.md"
        if description_path.exists():
            description = description_path.read_text()
        else:
            description = config.get("description", "")

        system_prompt_path = path_obj / "system_prompt.md"
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text().strip()
        else:
            system_prompt = config.get("system_prompt")

        required_tools = config.get("required_tools", [])
        execution_environment = config.get("execution_environment", "REPL")

        return Role(
            name=name,
            description=description,
            system_prompt=system_prompt,
            required_tools=required_tools,
            execution_environment=execution_environment
        )
