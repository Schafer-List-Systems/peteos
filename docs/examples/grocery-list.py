"""Example: Grocery List — agentic object interaction.

Usage: python grocery-list.py <backend-url>
"""

from enum import Enum
import asyncio
import sys

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool


_PRICES = {
    "Milk": 1.50,
    "Bread": 2.00,
    "Egg": 0.25,
    "Butter": 3.00,
}


class Grocery(Enum):
    MILK = "Milk"
    BREAD = "Bread"
    EGG = "Egg"
    BUTTER = "Butter"


class GroceryList(AgenticObjectBase):
    """You manage a grocery list. Read the current list with list_items,
    add items with add_item, and clear the list with clear."""

    def __init__(self):
        super().__init__()
        self._items: dict[Grocery, int] = {}

    @tool
    def list_items(self) -> list[tuple[str, float, int]]:
        """Return the current grocery list with prices."""
        return [
            (item.value, _PRICES[item.value], qty)
            for item, qty in self._items.items()
        ]

    @tool
    def add_item(self, item: Grocery, quantity: int) -> str:
        """Add items to the grocery list."""
        self._items[item] = self._items.get(item, 0) + quantity
        return f"Added {quantity} {item.value}s."

    @tool
    def clear(self) -> str:
        """Clear the grocery list."""
        self._items.clear()
        return "List cleared."


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    url = sys.argv[1]
    await ChatBotManager.add_backend("local", url)

    groceries = GroceryList()

    # Add items
    result = await groceries.invoke_agent("Add 2 milk and 3 eggs to the list.")
    print("After add:", result)

    # Check total cost
    result = await groceries.invoke_agent("How much will my shopping cost?")
    print("Cost:", result)

    # Clear
    result = await groceries.invoke_agent("I'm done shopping, clear the list.")
    print("After clear:", result)

    # Verify empty
    result = await groceries.invoke_agent("What's on the list?")
    print("List after clear:", result)


if __name__ == "__main__":
    asyncio.run(main())
