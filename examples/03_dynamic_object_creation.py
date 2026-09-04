#!/usr/bin/env python3
"""Dynamic Object Creation example from the OAP API documentation.

Allow agents to create new agentic objects dynamically and add them to
collections via sandboxed code execution.
"""

from dataclasses import dataclass
from peteos import AgenticObject, Error, agentic_object, tool


@agentic_object(allow_code_execution=True)
class InventoryItem(AgenticObject):
    """A single inventory item with name, quantity, and category."""

    def __init__(self, name: str = "New Item", quantity: int = 0, category: str = "uncategorized"):
        super().__init__()
        self._name = name
        self._quantity = quantity
        self._category = category

    @tool
    def get_name(self) -> str:
        """Get the item name."""
        return self._name

    @tool
    def set_name(self, name: str) -> None:
        """Set the item name."""
        self._name = name

    @tool
    def get_quantity(self) -> int:
        """Get the item quantity."""
        return self._quantity

    @tool
    def set_quantity(self, quantity: int) -> None:
        """Set the item quantity."""
        self._quantity = quantity

    @tool
    def get_category(self) -> str:
        """Get the item category."""
        return self._category

    @tool
    def set_category(self, category: str) -> None:
        """Set the item category."""
        self._category = category


@agentic_object(allow_code_execution=True, imports=[InventoryItem])
class InventoryManager(AgenticObject):
    """Manages a collection of inventory items. Uses sandboxed code to create new items."""

    def __init__(self):
        super().__init__()
        self._items: list[InventoryItem] = []

    @tool
    def get_items(self) -> list[InventoryItem]:
        """Get all items in the inventory."""
        return self._items

    @tool
    def get_item_count(self) -> int:
        """Get the number of items."""
        return len(self._items)

    @tool
    def list_items_summary(self) -> str:
        """List all items as a summary string."""
        lines = []
        for i, item in enumerate(self._items):
            lines.append(f"  {i}: {item.get_name()} (qty={item.get_quantity()}, cat={item.get_category()})")
        return "\n".join(lines) if lines else "  (empty)"


@dataclass
class SetupResult:
    item_count: int
    items: list[str]


async def main():
    """Set up an Agent and invoke it to create inventory items."""
    # --- Create OAP object (Agent is auto-created in __init__) ---
    manager = InventoryManager()

    # --- Invoke the agent ---
    try:
        result = await manager.invoke_agent(
            prompt=(
                "Create 3 items: 'Widget A' (qty=10, cat=cat-0), "
                "'Widget B' (qty=0, cat=cat-1), 'Gadget C' (qty=20, cat=cat-2). "
                "Use sandboxed code to instantiate InventoryItem objects and "
                "append them to self._items."
            ),
            output_schema=SetupResult,
            persistent_thread_id="setup-001",
        )
        if isinstance(result, Error):
            print(f"Error: {result.message}")
        else:
            print(f"Result: {result}")
    except Exception as e:
        print(f"API failure: {e!r}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
