# Getting Started

Welcome to Peteos.
This section covers the essentials to get you up and running.
First, you will learn how to install Peteos into your project.
Then, you will set up the LLM backend and chatbot environment.
Finally, you will see how to set up an agentic object and use it.


## Installation

Create a Python virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Clone the repository, then install Peteos:

```bash
git clone https://github.com/yourusername/peteos.git
cd peteos
pip install .
```

This installs Peteos as a package so you can `import peteos` from your own code.

Peteos requires Python 3.10 or later. Its core dependencies are `aiohttp`, `httpx`, and `tiktoken` (for token counting). If you plan to use camera-related agentic objects, install the optional extras:

```bash
pip install ".[camera]"
```

For development, install the test and lint dependencies:

```bash
pip install -e ".[dev]"
```

The `-e` flag installs Peteos in editable (development) mode, so changes to the source are reflected immediately without needing to reinstall.


## Setting Up the Environment

> _[Placeholder: Install instructions._]

## Setting Up the Environment

Configure the LLM backend using the `ChatBotManager` singleton:

```python
from peteos.chatbot.manager import ChatBotManager

await ChatBotManager.add_backend("local", "http://localhost:8000")
```

This registers an LLM backend that the agent harness uses during invocation.
The backend URL is passed at runtime — no hardcoded endpoints.

Additional options such as `streaming` and `max_tokens` can be configured per backend.
A single backend may expose multiple models, and the agent harness automatically selects one if the backend supports more than one.
Model patterns can be used to restrict which models are considered, and in case of ambiguity the harness picks a default. Details on configuration, model selection, and the full set of options are covered in the **[reference](./reference/api-overview.md)**.

---

## Hello World

An agentic class is a class that derives from `AgenticObjectBase`.
Behind the scenes, an agent operates on the object.
The class docstring serves as the agent's system prompt.

```python
class HelloPete(AgenticObjectBase):
    """You are Pete, a helpful assistant."""
```

That is all it takes to create an agentic class.
Invoke it directly:

```python
pete = HelloPete()
result = await pete.invoke_agent("Hello, what's your name?")
print(result)  # My name is Pete.
```

The agent reasons about your prompt and responds directly.
This is the simplest way to use an agentic object: create the instance, invoke the agent, get a response.

### Structured Output

You can force structured output by passing an `output_schema`:

```python
result = await pete.invoke_agent(
    prompt="What is your name and your most significant character trait?",
    output_schema=dict[str, str],
)
print(result)  # {"name": "Pete", "character": "helpful"}
```

The `output_schema` parameter tells the agent to return structured output.
No additional prompt engineering is needed — the agent adapts to the schema.
Output schemas can be primitives (`str`, `int`, `float`, `bool`), `list`, `dict`, `Enum`, or `dataclass`.

### Object Interaction

Agentic objects maintain their own state and reason about it. The agent reads and modifies the object using tools:

```python
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
```

The agent reasons about prompts and calls tools to interact with the object's state:

```python
groceries = GroceryList()
await groceries.invoke_agent("Add 2 milk and 3 eggs to the list.")
await groceries.invoke_agent("How much will my shopping cost?")
# 1.50 * 2 + 0.25 * 3 = 3.75
```

The docstring of a tool is the description the agent sees when deciding which tool to call.
Providing detailed descriptions — including what the arguments represent — helps the agent call tools correctly on the first try.
Returning a value makes the result visible to the agent; if the function returns `None`, no result is communicated.

### Sandboxed Code Execution

Enable the agent to run Python code in a restricted sandbox by decorating the class with `@agentic_object(allow_code_execution=True)`.
This enables a hidden tool called `python_exec` that the agent can call.
Code executed in the sandbox can access the agentic object through the `this` variable:

```python
@agentic_object(allow_code_execution=True)
class FibonacciSeries(AgenticObjectBase):
    """You are a helpful assistant."""
```

```python
seq = FibonacciSeries()
result = await seq.invoke_agent(
    f"Compute the sequence where each element is the sum of the squares of its two predecessors."
    f" Start with 0, 1. And compute the 10-th element.",
    output_schema=int,
)
print(result)
```

The sandbox provides pure functions and builtins but blocks file I/O, network access, and dangerous operations.

The `@agentic_object` decorator offers additional options such as `imports` to add modules to the sandbox, and `import_aliases` for module aliases. Providing more imports adds functionality but also increases the attack surface.

Beyond single objects, agentic systems can compose into hierarchies of agents and sub-agents for more complex tasks. Within the sandbox, the Python code can access the agentic object directly as if it were a member function. See the **[reference](./reference/api-overview.md)** for details.

### Example: Stock Portfolio Analyzer

A more complex example that combines all features — tools, sandboxed code execution, and structured output:

```python
import math
import random
import statistics
from dataclasses import dataclass
from enum import Enum

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool


class Sector(Enum):
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCE = "Finance"
    ENERGY = "Energy"
    CONSUMER = "Consumer"


@dataclass
class Holding:
    ticker: str
    shares: int
    sector: Sector


@agentic_object(allow_code_execution=True, imports=[math, random, statistics])
class StockPortfolioAnalyzer(AgenticObjectBase):
    """You are a stock portfolio analyzer. You manage a portfolio of stock
    holdings, current prices, and historical returns. Use add_holding and
    set_price to manage data, list_holdings to inspect the portfolio, and
    get_returns to view historical returns. The sandbox can use math and
    statistics to calculate performance metrics."""

    def __init__(self):
        super().__init__()
        self._holdings: list[Holding] = []
        self._prices: dict[str, float] = {}
        self._returns: list[float] = []

    @tool
    def add_holding(self, ticker: str, shares: int, sector: Sector) -> str:
        """Add a stock holding with ticker, number of shares, and sector."""
        self._holdings.append(Holding(ticker=ticker, shares=shares, sector=sector))
        return f"Added holding: {ticker} — {shares} shares ({sector.value})."

    @tool
    def set_price(self, ticker: str, price: float) -> str:
        """Set the current price for a stock ticker."""
        self._prices[ticker] = price
        return f"Price set for {ticker}: {price:.2f}."

    @tool
    def set_returns(self, returns: list[float]) -> str:
        """Set historical returns for a stock (list of daily percentage returns)."""
        self._returns = returns
        return f"Recorded {len(returns)} return values."

    @tool
    def list_holdings(self) -> list[dict]:
        """Return all holdings with current prices."""
        result = []
        for h in self._holdings:
            price = self._prices.get(h.ticker, 0.0)
            result.append({
                "ticker": h.ticker,
                "shares": h.shares,
                "sector": h.sector.value,
                "price": price,
                "value": round(h.shares * price, 2),
            })
        return result

    @tool
    def get_returns(self) -> list[float]:
        """Return the historical returns (list of daily percentage returns)."""
        return list(self._returns)


@dataclass
class PerformanceMetrics:
    total_value: float
    avg_return: float
    std_deviation: float
    sharpe_ratio: float


@dataclass
class SectorAllocation:
    sectors: list[dict[str, float]]
    recommended_allocation: dict[str, float]
    rebalance_note: str
```

```python
portfolio = StockPortfolioAnalyzer()
await portfolio.invoke_agent(
    "Add holdings: AAPL 50 shares (Technology), GOOGL 20 shares (Technology), "
    "JNJ 30 shares (Healthcare), XOM 40 shares (Energy)."
)
await portfolio.invoke_agent(
    "Set prices: AAPL 175.50, GOOGL 141.80, JNJ 156.30, XOM 104.20."
)
await portfolio.invoke_agent(
    "Set returns: [0.5, -0.2, 0.8, -0.4, 0.3, 0.1, -0.6, 0.9, 0.2, -0.1]."
)

result = await portfolio.invoke_agent(
    "Calculate total portfolio value and performance metrics (average return, "
    "standard deviation, Sharpe ratio).",
    output_schema=PerformanceMetrics,
)
print("Performance metrics:", result)

result = await portfolio.invoke_agent(
    "Analyze the sector allocation and suggest a recommended diversification.",
    output_schema=SectorAllocation,
)
print("Sector allocation:", result)
```

The agent reasons about the portfolio, calls its tools to inspect data, and uses the sandbox to perform calculations — all while returning structured output compatible with OOP workflows. See the **[full example](./examples/stock-portfolio-analyzer.py)** for a runnable version.

### Where to Go Next

- See the **[concepts](./concepts/agentic-objects.md)** section to understand how agentic objects work under the hood
- Explore the **[guide](./guide/creating-agents.md)** for topic-based walkthroughs
- Look at the **[agentic-objects module](https://github.com/schpe/peetos/tree/main/peteos/agentic_objects)** for real-world examples, including [BashWorkspace](https://github.com/schpe/peetos/tree/main/peteos/agentic_objects/bash_workspace.py) and [PDFTranscriber](https://github.com/schpe/peetos/tree/main/peteos/agentic_objects/pdf_transcriber.py)
