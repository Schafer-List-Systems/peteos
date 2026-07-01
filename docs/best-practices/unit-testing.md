# Unit Testing

Test agentic objects in two layers: deterministic tests first, non-deterministic tests second. This gives fast, reliable feedback while the agent system is being built up.

## Why

Non-deterministic behavior makes testing harder — a single failure may just be a bad agent turn, not a broken implementation. By testing deterministic code first, you verify the foundation. When non-deterministic tests fail, you know exactly which layer is at fault.

## How

### Layer 1: Deterministic Tests

Test every method that does neither directly nor indirectly call `invoke_agent`. These are standard unit tests — fast, deterministic, and easy to debug.

```python
def test_add_item():
    store = Inventory()
    result = store.add_item("laptop", 2)
    assert result == "Added 2 laptops."
    assert store.get_inventory() == [("laptop", 2)]
```

If a method modifies state or performs calculations, test both the return value and the resulting state.

```python
def test_set_price_updates_value():
    item = Product("widget", price=10.0)
    item.set_price(15.0)
    assert item.current_price == 15.0
```

### Layer 2: Agentic Tests

Test the agent's behavior with `invoke_agent()`, accepting non-deterministic results. Use `BenchmarkRunner` to run each test across multiple iterations and assert a minimum success rate.

```python
from peteos.oap.benchmark import BenchmarkRunner, BenchmarkRow

async def test_summarize_sentiment(row: BenchmarkRow) -> bool:
    analyzer = SentimentAnalyzer()
    analyzer._documents = row.input_dimensions["documents"]
    result = await analyzer.invoke_agent(
        "Summarize the overall sentiment.",
        output_schema=Summary,
    )
    return isinstance(result, Summary) and result.text
```

Define test cases as dimensions and add an iteration dimension for Monte Carlo runs:

```python
runner = BenchmarkRunner(test_summarize_sentiment)
runner.add_dimension("documents", [
    ["Great product!", "Terrible service."],
    ["Absolutely love it!", "Five stars!"],
])
runner.add_dimension("run", range(5))
report = await runner.run()
summary = report.average(["success"])
assert summary["success"] >= 0.8, f"Accuracy {summary['success']:.0%} below 80% threshold"
```

The `BenchmarkRunner` executes each test-case + run combination, tracking running success rates and printing a final aggregate summary. The developer is responsible for defining the test function, setting dimensions, and choosing the threshold. The runner itself does not create objects or set thresholds.

## Key Principles

- **Test deterministic methods directly.** Every method that does not call `invoke_agent` should have a deterministic unit test.
- **Test agent invocations separately.** Use `invoke_agent()` only in the second layer.
- **Set thresholds, not absolutes.** Non-deterministic tests should pass when accuracy exceeds a minimum rate.
- **Build up gradually.** Verify the deterministic foundation before adding agent layers.

> See also: [Testing](../concepts/testing.md)
