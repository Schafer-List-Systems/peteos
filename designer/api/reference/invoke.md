# `AgenticObject.invoke()` Method

> **Module**: `peteos.AgenticObject`
> **Purpose**: Invoke a sub-agent on another agentic object.

## Signature

```python
class AgenticObject:
    def invoke(
        self,
        target: "AgenticObject",
        prompt: str,
        output_schema: type | None = None,
        persistent: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """Invoke a sub-agent on a target object."""
        ...
```

## Description

Called from within sandboxed Python code. Invokes the global `invoke_agent()` on the target object with its own context-isolated execution. Available on every `AgenticObject` subclass — not gated by the calling object's decorator flags.

## Concurrency

`invoke()` and `invoke_agent()` are **serialized per `AgenticObject` instance** using an internal `threading.Lock`. Only one invocation runs at a time per object. This protects the object's member variables from race conditions caused by interleaved tool calls.

See also: [`@agentic_object`](agentic_object.md#members) (acquire/release members)

## Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `target` | `AgenticObject` | required | The sub-object to invoke the sub-agent on |
| `prompt` | `str` | required | Task description for the sub-agent |
| `output_schema` | `type` | `None` → `str` | Expected return type for the sub-agent |
| `persistent` | `bool` | `False` | If `True`, inherit the parent's thread ID; if `False`, use a non-persistent thread that is destroyed immediately after the call |
| `timeout` | `float \| None` | `None` | Maximum seconds to wait for the invocation lock on the target object. `None` = block indefinitely. Raises `TimeoutError` if the lock is not acquired in time. |

## Return Value

| `output_schema` | Return type |
|---|---|
| `None` (default) | `str` — the sub-agent's plain text response |
| `SomeClass` (dataclass, etc.) | `SomeClass` instance — structured result from the sub-agent |
| Sub-agent fails task | `Error` object (not an exception) |
| Sub-agent API breaks | Exception raised |

## Prerequisites

- The **target** object must have `@agentic_object(invoke_sub_agents=True)`
- The calling code must be sandboxed (enabled by `allow_code_execution=True` on the calling object's class)
- If the target's `invoke_sub_agents` is `False` (default), `invoke()` returns an `Error`

## Thread ID Inheritance

When `persistent=False` (default): the sub-agent runs with a non-persistent thread. The thread is destroyed immediately after the call returns.

When `persistent=True`: the sub-agent inherits the parent's thread ID from the harness execution context. The session persists on the target object and can be continued by subsequent calls with the same thread ID.

**Restriction:** once `persistent=True`, the thread ID is fixed — it cannot be changed during the sub-agent's lifetime.

```
Parent thread: "parent-tick"
  └─→ Sub-agent on Child (persistent=True) inherits thread_id="parent-tick"
  └─→ Sub-agent on Child (persistent=False) gets a fresh thread
```

## Usage

```python
@agentic_object(allow_code_execution=True)
class Parent(AgenticObject):
    def __init__(self):
        self._children = [Child("Alice"), Child("Bob")]

    @tool
    def get_children(self) -> list[Child]:
        return self._children


@agentic_object(invoke_sub_agents=True)
class Child(AgenticObject):
    @tool
    def get_state(self) -> dict:
        return {"hunger": 70, "tiredness": 30}

    @tool
    def eat(self) -> str:
        return "Fed."


@dataclass
class Answer:
    response: str


# Sandbox code from the parent agent:
for child in self.get_children():
    answer = self.invoke(child, "Do you want to play?", output_schema=Answer)
    # answer → Answer(response="Yes!") or Answer(response="No, I'm tired.")
```

## Flow

1. `invoke()` checks `target.invoke_sub_agents` — `True` proceeds, `False` returns `Error`
2. If `persistent=True`, reads the parent's thread ID from the harness execution context
3. If `persistent=False`, uses `thread_id=None` (non-persistent thread, destroyed after call)
4. Calls `invoke_agent(target=target, prompt=prompt, output_schema=output_schema, thread_id=...)`
5. Sub-agent runs on `target` with its own isolated context window
6. Result (structured output, `Error`, or exception) is returned to the sandbox code