# Sub-Object Invocation

Invoke sub-agents on nested agentic objects for context-isolated reasoning.

## Example

```python
from dataclasses import dataclass
from peteos import AgenticObjectBase, tool, invoke_agent, agentic_object

@dataclass
class PriceData:
    value: float
    currency: str

@agentic_object(invoke_sub_agents=True)
class PriceRecord(AgenticObjectBase):
    def __init__(self, raw_value: str = "100 USD"):
        self._raw_value = raw_value

    @tool
    def get_raw_value(self) -> str:
        """Get the raw price string."""
        return self._raw_value

    @tool
    def parse_price(self) -> float:
        """Extract numeric value from raw_value."""
        ...

class InventoryItem(AgenticObjectBase):
    def __init__(self, name: str, price: PriceRecord):
        self._name = name
        self._price = price

    @tool
    def get_name(self) -> str:
        return self._name

    @tool
    def get_price(self) -> PriceRecord:
        return self._price

@agentic_object(allow_code_execution=True)
class InventoryManager(AgenticObjectBase):
    def __init__(self):
        self._items = [
            InventoryItem(name="Widget A", price=PriceRecord(raw_value="100 USD")),
            InventoryItem(name="Widget B", price=PriceRecord(raw_value="50 EUR")),
        ]

    @tool
    def get_items(self) -> list[InventoryItem]:
        """Get the items list."""
        return self._items

result = invoke_agent(
    InventoryManager(),
    prompt="Parse all prices across all items using sandboxed code "
           "(self.invoke(item.get_price(), 'Parse this price', output_schema=PriceData)).",
    output_schema=list[PriceData],
)
```

**Expected output:**

```python
[
    PriceData(value=100.0, currency="USD"),
    PriceData(value=50.0, currency="EUR"),
]
```

## Why sub-agents?

Each sub-agent invocation gets its own isolated context window. The parent agent sees the results — not the sub-agent's reasoning. This keeps the parent's context small even when iterating over many items.
