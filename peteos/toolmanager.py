import inspect
from typing import Any, Callable, Optional


class Tool:
    """Represents a tool with its properties."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[dict] = None
    ):
        """
        Initialize Tool.

        Args:
            name: The name of the tool.
            description: Description of what the tool does.
            func: The callable that implements the tool.
            parameters: Tool parameters schema (if provided).
        """
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or self._extract_parameters(func)

    @staticmethod
    def _extract_parameters(func: Callable) -> dict:
        """Extract parameter information from a callable."""
        signature = inspect.signature(func)
        parameters = {}

        for param_name, param in signature.parameters.items():
            param_info: dict = {}

            # Get type information
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation)
            else:
                param_info["type"] = "any"

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                param_info["required"] = True
            else:
                param_info["required"] = False
                param_info["default"] = param.default

            parameters[param_name] = param_info

        return parameters

    @classmethod
    def from_callable(cls, func: Callable) -> "Tool":
        """Create a Tool from a callable, extracting name, description, and parameters."""
        name = func.__name__
        description = func.__doc__ or ""
        return cls(name=name, description=description, func=func)

    def __call__(self, **kwargs) -> Any:
        """Call the tool with provided arguments."""
        return self.func(**kwargs)


class ToolManager:
    """Manages tools for the execution environment."""

    def __init__(self):
        """Initialize ToolManager."""
        self._tools: dict[str, Tool] = {}

    def register_tool(
        self,
        tool: Optional[Tool] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[dict] = None,
        func: Optional[Callable] = None
    ) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register (preferred method).
            name: Name of the tool (required if func is provided).
            description: Description of the tool (required if func is provided).
            parameters: Tool parameters schema (optional).
            func: Callable to register as a tool (will extract name, description from docstring if not provided).
        """
        if tool is not None:
            # Register by Tool instance
            self._tools[tool.name] = tool
        elif func is not None:
            # Register by callable
            if name is None or description is None:
                # Extract from callable
                extracted_tool = Tool.from_callable(func)
                name = extracted_tool.name
                description = extracted_tool.description

            tool = Tool(name=name, description=description, func=func, parameters=parameters)
            self._tools[tool.name] = tool
        else:
            raise ValueError("Either 'tool' or 'func' with 'name' and 'description' must be provided")

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: The name of the tool.

        Returns:
            The Tool instance, or None if not found.
        """
        return self._tools.get(name)
