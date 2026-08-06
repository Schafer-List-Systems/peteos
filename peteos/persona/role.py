import json
from pathlib import Path
from typing import Callable, Optional


class Role:
    """A role with name, description, optional system prompt, and configuration."""

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        system_prompt_hooks: Optional[list[Callable[[], str]]] = None,
        required_tools: Optional[list[str]] = None,
        execution_environment: str = "REPL",
        model: str = ".*",
        auto_approve_tools: Optional[list[str]] = None,
        tool_filter: Optional[list[str]] = None,
        behavior_policy: str = "responsive",
        max_truncation_retries: int = 2
    ):
        """
        Initialize Role.

        Args:
            name: The name of the role.
            description: Description of the role.
            system_prompt: Optional system prompt (static text from role definition).
            system_prompt_hooks: Optional callbacks invoked at session creation.
                Each returns a text string appended to the system prompt.
            required_tools: Optional list of tool names required by this role.
            execution_environment: Name of the execution environment (default: "REPL").
            model: Regex pattern to match model IDs (default: ".*" matches any model).
            auto_approve_tools: Optional list of tool names auto-approved for this role.
            tool_filter: Optional list of regex patterns. Only tools whose names match
                any pattern are visible to this role.
            behavior_policy: Agent behavior strategy. "responsive" yields on text output,
                "continuous" keeps looping until yield_back is called.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.system_prompt_hooks = system_prompt_hooks if system_prompt_hooks is not None else []
        self.required_tools = required_tools if required_tools is not None else []
        self.execution_environment = execution_environment
        self.model = model
        self.auto_approve_tools = auto_approve_tools if auto_approve_tools is not None else []
        self.tool_filter = tool_filter if tool_filter is not None else []
        self.behavior_policy = behavior_policy
        self.max_truncation_retries = max_truncation_retries

    def __copy__(self) -> "Role":
        import copy
        return Role(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            system_prompt_hooks=list(self.system_prompt_hooks),
            required_tools=list(self.required_tools),
            execution_environment=self.execution_environment,
            model=self.model,
            auto_approve_tools=list(self.auto_approve_tools),
            tool_filter=list(self.tool_filter),
            behavior_policy=self.behavior_policy,
            max_truncation_retries=self.max_truncation_retries,
        )

    def add_system_prompt_hook(self, hook: Callable[[], str]) -> None:
        """Add a hook that returns a dynamic fragment for the system prompt."""
        self.system_prompt_hooks.append(hook)

    @property
    def _all_hooks(self) -> list[Callable[[], str]]:
        """Return hooks including static system_prompt as a lambda."""
        hooks = list(self.system_prompt_hooks)
        if self.system_prompt:
            hooks.append(lambda sp=self.system_prompt: sp)
        return hooks

    @staticmethod
    def load_from_dict(data: dict) -> "Role":
        """
        Creates a new role from a dictionary.

        Args:
            data: Dictionary with 'name', 'description', and optional 'system_prompt',
                  'required_tools', 'execution_environment', 'model' keys.

        Returns:
            A new Role instance.
        """
        name = data["name"]
        description = data.get("description")
        system_prompt = data.get("system_prompt")
        required_tools = data.get("required_tools", [])
        execution_environment = data.get("execution_environment", "REPL")
        model = data.get("model", ".*")
        auto_approve_tools = data.get("auto_approve_tools", [])
        tool_filter = data.get("tool_filter", [])
        behavior_policy = data.get("behavior_policy", "responsive")
        max_truncation_retries = data.get("max_truncation_retries", 2)
        return Role(
            name=name,
            description=description,
            system_prompt=system_prompt,
            required_tools=required_tools,
            execution_environment=execution_environment,
            model=model,
            auto_approve_tools=auto_approve_tools,
            tool_filter=tool_filter,
            behavior_policy=behavior_policy,
            max_truncation_retries=max_truncation_retries,
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

        Description is loaded from 'description.md' (or 'config.json.description' as fallback).
        System prompt is loaded from 'system_prompt.md' (or 'config.json.system_prompt' as fallback).
        Both fields are optional and default to None.
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
        description = None
        if description_path.exists():
            description = description_path.read_text()
        elif "description" in config:
            description = config.get("description")

        system_prompt_path = path_obj / "system_prompt.md"
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text().strip()
        else:
            system_prompt = config.get("system_prompt")

        required_tools = config.get("required_tools", [])
        execution_environment = config.get("execution_environment", "REPL")
        model = config.get("model", ".*")
        auto_approve_tools = config.get("auto_approve_tools", [])
        tool_filter = config.get("tool_filter", [])
        behavior_policy = config.get("behavior_policy", "responsive")
        max_truncation_retries = config.get("max_truncation_retries", 2)

        return Role(
            name=name,
            description=description,
            system_prompt=system_prompt,
            required_tools=required_tools,
            execution_environment=execution_environment,
            model=model,
            auto_approve_tools=auto_approve_tools,
            tool_filter=tool_filter,
            behavior_policy=behavior_policy,
            max_truncation_retries=max_truncation_retries,
        )
