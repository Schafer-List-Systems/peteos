"""Test for OAP tool-based agentic interaction via GroceryList."""

from enum import Enum

import pytest

from peteos.oap.agentic_object import AgenticObject
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


class GroceryList(AgenticObject):
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


async def test_grocery_list_add_and_member_state():
    groceries = GroceryList()

    result = await groceries.invoke_agent("Add 2 milk and 3 eggs to the list.")
    assert "2 Milk" in result or "2 milk" in result or "Added" in result
    assert groceries._items == {Grocery.MILK: 2, Grocery.EGG: 3}


async def test_grocery_list_cost_and_member_state():
    groceries = GroceryList()
    groceries._items = {Grocery.MILK: 2, Grocery.EGG: 3}

    result = await groceries.invoke_agent("How much will my shopping cost?")
    assert "3.75" in result
    assert groceries._items == {Grocery.MILK: 2, Grocery.EGG: 3}


async def test_grocery_list_clear_and_member_state():
    groceries = GroceryList()
    groceries._items = {Grocery.MILK: 2, Grocery.EGG: 3}

    result = await groceries.invoke_agent("I'm done shopping, clear the list.")
    assert "cleared" in result.lower()
    assert groceries._items == {}


async def test_grocery_list_empty_after_clear():
    groceries = GroceryList()
    groceries._items = {Grocery.MILK: 2, Grocery.EGG: 3}
    groceries._items.clear()

    result = await groceries.invoke_agent(
        "What's on the list?",
        output_schema=list[Grocery],
    )
    assert result == []
