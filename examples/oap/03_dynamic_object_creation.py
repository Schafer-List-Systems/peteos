#!/usr/bin/env python3
"""Dynamic Object Creation example from the OAP API documentation.

Allow agents to create new agentic objects dynamically and add them to
collections via sandboxed code execution.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/oap/03_dynamic_object_creation.py

Configure your LLM backend before running:
    await chatbot_manager.add_backend("name", "http://your-backend:port")
"""

from dataclasses import dataclass

from peteos import AgenticObjectBase, Error, agentic_object, tool
from peteos.oap import invoke


@agentic_object(allow_code_execution=True)
class InventoryItem(AgenticObjectBase):
    """A single inventory item with name, quantity, and category."""

    def __init__(self, name: str = "New Item", quantity: int = 0, category: str = "uncategorized"):
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


@agentic_object(allow_code_execution=True)
class InventoryManager(AgenticObjectBase):
    """Manages a collection of inventory items. Uses sandboxed code to create new items."""

    def __init__(self):
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
    # --- Set up peteos components ---
    from peteos.agent import Agent
    from peteos.chatbot.manager import ChatBotManager
    from peteos.role import Role
    from peteos.rolemanager import RoleManager
    from peteos.toolmanager import ToolManager

    role_manager = RoleManager()
    chatbot_manager = ChatBotManager(timeout=None)
    tool_manager = ToolManager()

    # Configure your LLM backend here:
    # await chatbot_manager.add_backend("name", "http://your-backend:port")

    # --- Create Agent and attach it to the OAP object ---
    agent = Agent(role_manager, chatbot_manager, tool_manager)

    manager = InventoryManager()
    manager.agent = agent

    # --- Invoke the agent ---
    try:
        result = await invoke(
            manager,
            prompt=(
                "Create 3 items: 'Widget A' (qty=10, cat=cat-0), "
                "'Widget B' (qty=0, cat=cat-1), 'Gadget C' (qty=20, cat=cat-2). "
                "Use sandboxed code to instantiate InventoryItem objects and "
                "append them to self._items."
            ),
            output_schema=SetupResult,
            thread_id="setup-001",
        )
        print(f"Success: {result['success']}")
        print(f"Result: {result['result']}")
        print(f"Thread ID: {result['thread_id']}")
    except Exception as e:
        print(f"API failure: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
