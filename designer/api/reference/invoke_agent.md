# `invoke_agent()` Function

> **Module**: `peteos.invoke_agent`
> **Purpose**: Main entry point for agent-driven object interaction.

## Signature

```python
def invoke_agent(
    object: AgenticObjectBase,
    prompt: str = "",
    output_schema: type | None = None,
    thread_id: str | None = None,
    timeout: float | None = None,
) -> Any:
    """Run an agent on an object. Returns structured output, Error, or raises Exception."""
    ...
```

## Description

Starts an agent that reasons over the object's `@tool` methods and works toward the given `prompt`. Returns a structured result matching `output_schema`, an `Error` object on task failure, or raises an exception on API failure.

## Concurrency

`invoke_agent()` and `AgenticObjectBase.invoke()` are **serialized per `AgenticObjectBase` instance** using an internal `threading.Lock`. Only one invocation runs at a time per object. This protects the object's member variables from race conditions caused by interleaved tool calls.

See also: [`@agentic_object`](agentic_object.md#members) (acquire/release members)

## Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `object` | `AgenticObjectBase` | required | The root object the agent will work on |
| `prompt` | `str` | `""` | Task description for the agent |
| `output_schema` | `type` | `None` → `str` | Expected return type (dataclass, TypedDict, etc.) |
| `thread_id` | `str \| None` | `None` → non-persistent | Persistent session ID. Omitted for single-shot calls |
| `timeout` | `float \| None` | `None` | Maximum seconds to wait for the invocation lock. `None` = block indefinitely. Raises `TimeoutError` if the lock is not acquired in time. |

## Return Value Semantics

`invoke_agent` returns **exactly one of**:

### Structured output (returned value)
When `output_schema` is provided and the agent satisfies the request:

```python
@dataclass
class Report:
    items: list[str]

result = invoke_agent(manager, prompt="List all items", output_schema=Report)
# result → Report(items=["Flour", "Sugar"])
```

### Error object (returned value, not raised)
When the agent worked but could not satisfy the request:

```python
result = invoke_agent(manager, prompt="Do something impossible", output_schema=Report)
# result → Error(message="Could not determine which items are low stock...")
```

### Exception (raised, not returned)
When the agent API itself failed — internal workflow broke:

```python
invoke_agent(manager, prompt="...")
# → RuntimeError("Tool schema generation failed")
```

## Thread Behavior

| `thread_id` | Behavior |
|---|---|
| `None` (default) | Non-persistent — fresh context each call, history discarded immediately |
| `"session1"` | Persistent — history preserved across calls with the same ID |

```python
# Non-persistent: no memory between calls
result1 = invoke_agent(obj, prompt="My favorite color is blue.")
result2 = invoke_agent(obj, prompt="What is my favorite color?")
# result2 → Error("I don't have that information.")

# Persistent: shared history
result1 = invoke_agent(obj, prompt="My favorite color is blue.", thread_id="colors")
result2 = invoke_agent(obj, prompt="What is my favorite color?", thread_id="colors")
# result2 → Error(response="blue")
```

## What Happens

1. `invoke_agent` reflects on the object, collecting all `@tool` decorated methods
2. It builds a tool schema (name, description, parameters, return types)
3. It starts the agent with the `prompt` and the object's interface
4. The agent reasons, calls tools, reads/writes state via getters/setters, executes sandboxed code
5. It returns structured output matching `output_schema`, or an `Error` object on failure
6. Exceptions are raised only for API-level failures, not task-level failures