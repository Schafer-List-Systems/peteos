# Testing

Testing agentic systems in the sOAP paradigm follows familiar patterns from classical OOP, with adaptations for the non-deterministic nature of AI-generated responses.

## Unit Testing Tools

Developers write tests for agentic classes just as they would for classical OOP classes — testing the tools (methods decorated with [`@tool`](../reference/decorator-args.md#tool)) that are part of the object's public interface.
Since these methods are regular Python functions, standard unit tests work directly:

```python
def test_add_item():
    groceries = GroceryList()
    result = groceries.add_item(Grocery.MILK, 2)
    assert "Added" in result
```

This includes testing any methods that do not involve agent invocation, making it straightforward to verify deterministic code paths.

## Monte Carlo Testing for Agentic Behavior

Testing agentic behavior is fundamentally different because LLM responses are non-deterministic.
A single run may succeed or fail due to reasoning quality, even when tool definitions and prompts are correct.

The sOAP paradigm addresses this with **Monte Carlo testing** using `BenchmarkRunner`.
Tests are defined as functions that return `True` or `False` for each run, and multiple iterations are executed for each test case:

```python
from peteos.oap.benchmark import BenchmarkRunner, BenchmarkRow

test_cases = [
    (True, "The sky is blue", "The color of the world's ceiling"),
    (True, "Water boils at 100 degrees Celsius", "Boiling temperature of water"),
    (False, "The sky is blue", "green"),
    (False, "Photosynthesis converts sunlight", "nuclear energy"),
]

async def test_fn(row: BenchmarkRow) -> bool:
    expected, text, substring = row.input_dimensions["test_case"]
    comp = AgenticStringComparator.instance()
    result = await comp.contains(text, substring)
    return result == expected

runner = BenchmarkRunner(test_fn)
runner.add_dimension("test_case", test_cases)
runner.add_dimension("run", range(5))

report = await runner.run()
summary = report.average(["success"])
assert summary["success"] >= 0.8, f"contains accuracy: {summary}"
```

A test defines:
- A test function (`test_fn`) that takes a `BenchmarkRow` and returns `True`/`False`.
- Dimensions: input cases and iteration counts. The runner creates a Cartesian product of all dimensions, executing the test once per combination.
- An assertion on `report.average(["success"])` — the developer sets the threshold (e.g., 80% success rate).

The `BenchmarkRunner` itself does not set thresholds, create test objects, count tokens, or aggregate metrics.
These are all the developer's responsibility.
The runner simply executes and tracks running success rates, printing per-row progress and a final summary.

## Benchmarking Infrastructure

Peteos provides a `BenchmarkRunner` utility that executes test functions across multiple iterations, tracking running success rates and producing a summary with per-dimension breakdowns.
The runner does not set thresholds, count tokens, or create objects — these are the developer's responsibility.

This approach allows developers to:
- **Test individual components** with classical unit tests (for deterministic tool code).
- **Test agentic behavior** with Monte Carlo tests (for non-deterministic agent interactions).
- **Fine-tune prompts** by iterating on system prompts and tool descriptions while measuring improvements quantitatively.
- **Compose systems** from verified components, knowing each building block meets its reliability threshold.

## Testing Strategy

A practical testing strategy for an agentic system combines both approaches:
1. Write classical unit tests for all tool methods.
2. Write Monte Carlo tests for each agent invocation path.
3. Use the threshold mechanism to set acceptable quality levels and detect regressions.
