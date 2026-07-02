# `@tool` Decorator

> **Module**: `peteos.tool`
> **Purpose**: Expose a method as an agent tool.

## Signature

```python
def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], Callable]:
    """Decorate a method to make it available to agents."""
    ...
```

## Description

Marks a method on an `AgenticObject` subclass as callable by agents. The agent's reasoning loop treats every `@tool` method as an available action.

## Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str \| None` | `None` → method name | Override the tool name exposed to the agent |
| `description` | `str \| None` | `None` → docstring | Override the tool description exposed to the agent |

## Usage

```python
from peteos import AgenticObject, tool

class InventoryManager(AgenticObject):
    def __init__(self):
        self._items = ["Flour", "Hammer", "Sugar"]

    @tool
    def get_items(self) -> list[str]:
        """Return the current items."""
        return self._items

    @tool(name="list_items", description="Return all inventory items for analysis.")
    def get_items_verbose(self) -> list[str]:
        return self._items

    @tool
    def set_items(self, items: list[str]) -> None:
        """Replace the items list."""
        self._items = items

    @tool
    def remove_item(self, index: int) -> str:
        """Remove item at index. Returns removed name."""
        return self._items.pop(index)

    def internal_audit(self) -> None:
        """Not decorated — not visible to agents."""
        ...
```

## Resolution

| Without parameters | With parameters |
|---|---|
| Name: method name (`get_items`) | Name: explicit `name` |
| Description: docstring | Description: explicit `description` |
| Parameters: method signature | Parameters: method signature |
| Return type: return type hint | Return type: return type hint |

Parameters and return types are always derived from the method signature. Only name and description can be overridden.

## Rules

- Must be applied to methods of classes that inherit from `AgenticObject`
- All `@tool` methods are collected via reflection when `invoke_agent()` is called
- Undecorated methods are invisible to agents — callable only from normal Python code
- Without parameters: name derived from method name, description from docstring
- With parameters: name and description use explicit values; signature is always from the method
- Parameters and return types are always derived from the method signature

## Read-Only vs Read-Write

| Pattern | Effect |
|---|---|
| `@tool` getter only | Agent can read the member variable |
| `@tool` getter + `@tool` setter | Agent can read and write the member variable |

The developer controls which member variables are mutable by omitting the setter.

## What the Agent Sees

For each `@tool` method, the agent receives:
- **Tool name**: explicit `name` if provided, otherwise method name
- **Description**: explicit `description` if provided, otherwise method docstring
- **Parameters**: derived from method signature
- **Return type**: derived from return type hint

The agent does not see the implementation — only the resolved name, description, signature, and return type.
