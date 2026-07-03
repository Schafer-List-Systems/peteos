# Composition

Agentic objects, built from [`AgenticObject`](../reference/agentic-object-base.md) and configured with [`@agentic_object`](../reference/decorator-args.md), can be composed through multi-inheritance to build more complex behavior from simpler building blocks.

## Base Agentic Class

The simplest agentic class derives directly from [`AgenticObject`](../reference/agentic-object-base.md):

```python
class SimpleAssistant(AgenticObject):
    """You are a helpful assistant."""
```

This is all it takes to create an agentic class.
Behind the scenes, an agent operates on the object.

## Multi-Inheritance Composition

A class can derive from multiple agentic classes to compose their behavior:

```python
class ComplexAgent(ToolA, ToolB, AgenticObject):
    """You are a composed agent with capabilities A and B."""
```

To enable multi-inheritance, each parent class in the hierarchy must also derive from `AgenticObject`.
The system determines which classes in the inheritance chain are agentic by checking whether they directly inherit from `AgenticObject`.

## System Prompt Construction

When composing multiple agentic classes, the agent's system prompt is built by concatenating the docstrings of each agentic parent class in the Method Resolution Order (MRO).
The most basic class's docstring (typically `AgenticObject`) comes first, and classes appearing later in the inheritance line are appended later in the combined prompt.

If no parent classes have docstrings, the system falls back to a generic template based on the concrete class name.

## Config Merging

Configuration from decorated agentic parents (set via the `@agentic_object` decorator) is also merged across the inheritance hierarchy:

- `imports` are unioned across all parents.
  See [`@agentic_object` decorator arguments](../reference/decorator-args.md) for the full list.
- `import_aliases` are merged, with later classes' aliases taking precedence.
- Boolean flags (`allow_code_execution`, `allow_media_access`, `invoke_sub_agents`) are combined with a logical OR.

This means composed agentic objects inherit both their behavior and their configuration from all parents.

## Concrete Example

Consider an NPC in a game that needs to both navigate an environment and interact with objects.
We build two focused agentic classes and compose them.

```python
from peteos.oap.base import AgenticObject
from peteos.oap.decorators import agentic_object, tool


@agentic_object(allow_code_execution=True)
class Navigator(AgenticObject):
    """You are the navigation specialist of this NPC. You understand
    maps, positions, and movement. Use move_to to travel to locations."""

    def __init__(self):
        super().__init__()
        self._position: tuple[int, int] = (0, 0)

    @tool
    def get_position(self) -> tuple[int, int]:
        """Return current position as (x, y)."""
        return self._position

    @tool
    def move_to(self, x: int, y: int) -> str:
        """Move to the specified coordinates."""
        self._position = (x, y)
        return f"Moved to ({x}, {y})."


@agentic_object(allow_code_execution=True)
class InventoryManager(AgenticObject):
    """You are the inventory specialist of this NPC. You understand
    what items are available and how they can be used."""

    def __init__(self):
        super().__init__()
        self._items: list[str] = []

    @tool
    def get_inventory(self) -> list[str]:
        """Return the current list of held items."""
        return self._items

    @tool
    def add_item(self, item: str) -> str:
        """Add an item to the inventory."""
        self._items.append(item)
        return f"Picked up: {item}."
```

The `Navigator` knows about positions and movement.
The `InventoryManager` knows about items.
Both are self-contained agentic objects with their own system prompts, tools, and state.

Now compose them into a single agent:

```python
@agentic_object(allow_code_execution=True)
class Adventurer(Navigator, InventoryManager, AgenticObject):
    """You are an adventurer navigating a dungeon. You can explore
    rooms, pick up items, and decide where to go next."""
```

The `Adventurer` inherits all tools from both parents (`get_position`, `move_to`, `get_inventory`, `add_item`) and all state.
The agent receives a combined system prompt:

```
You are the navigation specialist of this NPC.
You understand maps, positions, and movement.
Use move_to to travel to locations.

You are the inventory specialist of this NPC.
You understand what items are available and how they can be used.

You are an adventurer navigating a dungeon.
You can explore rooms, pick up items, and decide where to go next.
```

The most basic class (`AgenticObject`) contributes no prompt text (it has no docstring).
`Navigator` comes first in the MRO, `InventoryManager` second, then the `Adventurer` prompt itself comes last.
 Invoking this agent:

```python
adventurer = Adventurer()
result = await adventurer.invoke_agent(
    "Explore the dungeon. Pick up any items you find.",
    persistent_thread_id="adventure-1",
)
```

The agent reasons using the combined system prompt and calls tools from both parent classes:

```
[ToolCall: move_to(3, 5)]
[ToolCall: add_item("rusty sword")]
[ToolCall: get_inventory()]
Result: Picked up: rusty sword.

[ToolCall: move_to(4, 2)]
Result: Moved to (4, 2).
```
