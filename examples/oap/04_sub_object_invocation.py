#!/usr/bin/env python3
"""Sub-Object Invocation example from the OAP API documentation.

Invoke sub-agents on nested agentic objects for context-isolated reasoning.
Each sub-agent gets its own independent context window.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/oap/04_sub_object_invocation.py \
        http://localhost:8080
"""

import sys

from dataclasses import dataclass

from peteos import AgenticObjectBase, Error, agentic_object, tool


@dataclass
class PriceData:
    value: float
    currency: str


@agentic_object(invoke_sub_agents=True)
class PriceRecord(AgenticObjectBase):
    """A price record with a raw price string."""

    def __init__(self, raw_value: str = "100 USD"):
        super().__init__()
        self._raw_value = raw_value

    @tool
    def get_raw_value(self) -> str:
        """Get the raw price string."""
        return self._raw_value

    @tool
    def set_raw_value(self, value: str) -> None:
        """Set the raw price string."""
        self._raw_value = value

    @tool
    def parse_price(self) -> float:
        """Extract numeric value and currency from raw_value."""
        parts = self._raw_value.strip().split()
        value = float(parts[0])
        currency = parts[1] if len(parts) > 1 else "USD"
        return value


@agentic_object(allow_code_execution=True, imports=[PriceRecord])
class InventoryItem(AgenticObjectBase):
    """An inventory item with a price record."""

    def __init__(self, name: str, price: PriceRecord):
        super().__init__()
        self._name = name
        self._price = price

    @tool
    def get_name(self) -> str:
        """Get the item name."""
        return self._name

    @tool
    def get_price(self) -> PriceRecord:
        """Get the price record for this item."""
        return self._price


@agentic_object(allow_code_execution=True, imports=[InventoryItem, PriceRecord])
class InventoryManager(AgenticObjectBase):
    """Manages items with sub-agents for price parsing."""

    def __init__(self):
        super().__init__()
        self._items = [
            InventoryItem(name="Widget A", price=PriceRecord(raw_value="100 USD")),
            InventoryItem(name="Widget B", price=PriceRecord(raw_value="50 EUR")),
        ]

    @tool
    def get_items(self) -> list[InventoryItem]:
        """Get all items."""
        return self._items

    @tool
    def get_item_count(self) -> int:
        """Get the number of items."""
        return len(self._items)

    @tool
    def list_items(self) -> str:
        """List all items with their raw prices."""
        lines = []
        for item in self._items:
            lines.append(f"  {item.get_name()}: {item.get_price().get_raw_value()}")
        return "\n".join(lines)


async def main():
    """Set up an Agent and invoke it to parse prices via sub-agents."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <backend-url>")
        sys.exit(1)
    backend_url = sys.argv[1]

    # --- Set up peteos components ---
    from peteos.chatbot.manager import ChatBotManager

    chatbot_manager = ChatBotManager(timeout=None)
    await chatbot_manager.add_backend("local", backend_url)

    # --- Create OAP objects (Agents are auto-created in __init__) ---
    manager = InventoryManager()

    # --- Invoke the agent ---
    try:
        result = await manager.invoke_agent(
            prompt=(
                "Parse all prices across all items using sandboxed code. "
                "Iterate over self.get_items(), and for each item invoke "
                "self.invoke(item.get_price(), 'Parse this price', "
                "output_schema=PriceData) to get structured price data. "
                "Return a list of PriceData objects."
            ),
            output_schema=list[PriceData],
            persistent_thread_id="prices-001",
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
