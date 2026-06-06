# Dynamic Object Creation

Allow agents to create new agentic objects dynamically and add them to collections.

## Example

```python
from dataclasses import dataclass
from peteos import AgenticObjectBase, tool, invoke_agent, agentic_object

@agentic_object(allow_code_execution=True)
class InventoryItem(AgenticObjectBase):
    def __init__(self, name: str = "New Item", quantity: int = 0, category: str = "uncategorized"):
        self._name = name
        self._quantity = quantity
        self._category = category

    @tool
    def get_name(self) -> str:
        return self._name

    @tool
    def set_name(self, name: str) -> None:
        self._name = name

    @tool
    def get_quantity(self) -> int:
        return self._quantity

    @tool
    def set_quantity(self, quantity: int) -> None:
        self._quantity = quantity

    @tool
    def get_category(self) -> str:
        return self._category

    @tool
    def set_category(self, category: str) -> None:
        self._category = category

@agentic_object(allow_code_execution=True)
class InventoryManager(AgenticObjectBase):
    def __init__(self):
        self._items = []

    @tool
    def get_items(self) -> list[InventoryItem]:
        return self._items

    @tool
    def set_items(self, items: list[InventoryItem]) -> None:
        self._items = items

@dataclass
class SetupResult:
    item_count: int
    items: list[str]

result = invoke_agent(
    InventoryManager(),
    prompt="Create 3 items with quantities 0, 10, 20 in categories cat-0, cat-1, cat-2. "
           "Use sandboxed code to instantiate InventoryItem objects and append them to self._items.",
    output_schema=SetupResult,
    thread_id="setup-001",
)
```

**Expected output:**

```python
SetupResult(
    item_count=3,
    items=["Item 0", "Item 1", "Item 2"],
)
```

## Why an agent?

Creating objects in collections requires code — you can't do it with tool calls alone. The agent writes sandboxed Python to instantiate objects and manipulate collections. Without code execution, agentic objects can't be created at runtime.
