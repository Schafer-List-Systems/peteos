# Decorator Arguments

Two decorators configure agentic objects: `@agentic_object` sets agent capabilities per class, and `@tool` exposes methods to the agent.

## `@agentic_object`

```python
@agentic_object(
    imports=None,
    import_aliases=None,
    invoke_sub_agents=False,
    allow_code_execution=False,
    define_functions=False,
    role=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `imports` | `list[object]` | `None` | Modules available in the sandbox. Each entry must be an importable module. |
| `import_aliases` | `dict[str, str]` | `None` | Module name → alias mappings for the sandbox. Later aliases override earlier ones. |
| `invoke_sub_agents` | `bool` | `False` | Enable `invoke()` method for delegating tasks to other agentic objects. |
| `allow_code_execution` | `bool` | `False` | Enable the hidden `python_exec` tool in the sandbox. |
| `define_functions` | `bool` | `False` | Enable the `define_function` and `remove_function` tools, allowing an agentic object to dynamically create and unregister tools at runtime. Used by [`AdaptiveObject`](../concepts/adaptive-objects.md). |
| `role` | `str \| None` | `None` | Override the role name used for this agentic object. The canonical role (built from the class docstring and MRO) is still used, but the name field is set to this value for disambiguation. Can be overridden at runtime via `RoleManager`. |

When composing agentic classes via multi-inheritance, boolean flags are combined with logical OR and imports are unioned across the MRO.

**Example:**

```python
import math
import statistics

@agentic_object(allow_code_execution=True, imports=[math, statistics])
class Calculator(AgenticObject):
    """You compute statistics on provided data."""
```

## `@tool`

```python
@tool
def my_method(self, arg: str) -> int:
    """Description the agent sees when deciding whether to call this tool."""

@tool(name="custom_name")
def my_method(self, arg: str) -> int:
    """Method docstring is ignored; 'custom_name' is used as the tool name."""

@tool(description="Explicit description")
def my_method(self, arg: str) -> int:
    """Method docstring is ignored; explicit description is used."""
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | Method's `__name__` | Custom tool name shown to the agent. |
| `description` | `str \| None` | Method's `__doc__` stripped | Description the agent uses to decide whether to call the tool. |

The description is critical — the agent relies on it to determine when and how to call the tool. Provide detailed descriptions including what each argument represents.

**Example:**

```python
@tool
def add_item(self, item: str, quantity: int) -> str:
    """Add an item to the list. `item` is the product name, `quantity` is the number of units."""
    self._items.append((item, quantity))
    return f"Added {quantity} of {item}."
```
