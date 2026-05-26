# `Error` Class

> **Module**: `peteos.Error`
> **Purpose**: Task failure result — returned, not raised.

## Description

Represents a task-level failure where the agent worked but could not produce the desired result. Unlike Python exceptions, `Error` is a return value that the agent produces when it determines the request cannot be satisfied.

## Signature

```python
class Error:
    message: str
```

## Usage

```python
from peteos import invoke_agent

result = invoke_agent(obj, prompt="Do something impossible", output_schema=Result)
if isinstance(result, Error):
    print(f"Agent failed: {result.message}")
```

## When It Occurs

| Scenario | Example |
|---|---|
| Insufficient information | "Could not determine the price — no price data available" |
| Task outside tool capabilities | "Cannot calculate distance — no distance tool exists" |
| Contradictory requirements | "Cannot set both price to 10 and price to 20 simultaneously" |
| Semantic failure | "Could not classify items as groceries — insufficient context" |

## Distinction from Exceptions

| | `Error` | Exception |
|---|---|---|
| Meaning | Task failed | API failed |
| Who | The agent | The framework |
| When | Agent reasoned but couldn't satisfy the request | Internal workflow broke |
| Return | Returned as value | Raised with `raise` |
| Examples | "Could not determine...", "Insufficient info..." | RuntimeError, ToolSchemaError |

## Accessing the Error

```python
from peteos import Error

result = invoke_agent(obj, prompt="...", output_schema=MyResult)

if isinstance(result, Error):
    print(result.message)
    # "Could not determine which items are low stock — inventory data is inconsistent"
```
