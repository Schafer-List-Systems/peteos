# Error

```python
from peteos.oap import Error
```

`Error` is a value object returned by `invoke_agent` when the agent completed its work but could not produce the desired result. It is **not** an exception — it is returned alongside successful results.

```python
result = await obj.invoke_agent("some prompt")
if isinstance(result, Error):
    print(f"Agent failed: {result.message}")
else:
    print(f"Result: {result}")
```

## Constructor

```python
Error(message: str) -> None
```

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable explanation of why the agent could not produce the result. |

## Properties

| Property | Type | Description |
|---|---|---|
| `message` | `str` | The error explanation. |

## Equality

```python
Error("something broke") == Error("something broke")  # True
```

Two `Error` instances are equal when their messages match.
