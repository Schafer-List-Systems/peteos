# AgenticObjectBase

```python
from peteos.oap.base import AgenticObjectBase
```

Base class for all Object-Agentic Programming objects. Deriving from it gives every instance its own thinking agent. Configure class-level behavior with the [`@agentic_object` decorator](./decorator-args.md).

## Constructor

```python
class MyObject(AgenticObjectBase):
    def __init__(self):
        super().__init__()
        # ... your state ...
```

Do not override `__init__` without calling `super().__init__()` first. After that, initialize your own state as usual.

## Instance Methods

### `invoke_agent()`

```python
async def invoke_agent(
    prompt: str,
    output_schema: type | None = None,
    persistent_thread_id: str | None = None,
    timeout: float | None = None,
    image: str | None = None,
) -> Any
```

Invoke this object's agent with a text prompt.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | — | Task description for the agent. |
| `output_schema` | `type \| None` | `None` | Expected return type (dataclass, Enum, etc.). |
| `persistent_thread_id` | `str \| None` | `None` | If set, reuses or creates a persistent session keyed by this ID. |
| `timeout` | `float \| None` | `None` | Maximum seconds to wait for the invocation lock. Also used as a timeout for the agent loop. |
| `image` | `str \| None` | `None` | Optional local file path or HTTP(S) URL to attach an image. |

**Returns:** Structured output or `Error` object.

**Raises:** `TimeoutError` if the invocation lock is not acquired or the agent loop exceeds `timeout`.

### `invoke()`

```python
async def invoke(
    target: AgenticObjectBase,
    prompt: str,
    output_schema: type | None = None,
    persistent: bool = False,
    timeout: float | None = None,
) -> Any
```

Invoke a sub-agent on a target agentic object. Only works when `invoke_sub_agents=True` on the [`@agentic_object` decorator](./decorator-args.md).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target` | `AgenticObjectBase` | — | The sub-object to invoke. |
| `prompt` | `str` | — | Task description for the sub-agent. |
| `output_schema` | `type \| None` | `None` | Expected return type. |
| `persistent` | `bool` | `False` | If True, inherit the parent's thread ID. |
| `timeout` | `float \| None` | `None` | Maximum seconds to wait for the lock on the target. |

**Returns:** Structured output or `Error` object.

### `acquire()` / `release()`

```python
def acquire(timeout: float | None = None) -> None
def release() -> None
```

Acquire and release the invocation lock. Serializes concurrent `invoke_agent()` calls on the same object to prevent race conditions.

| Parameter | Type | Description |
|---|---|---|
| `timeout` | `float \| None` | Maximum seconds to wait. `None` = block indefinitely. |

**Raises:** `TimeoutError` if the lock is not acquired within `timeout`.
