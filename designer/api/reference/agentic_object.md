# `@agentic_object` Decorator

> **Module**: `peteos.agentic_object`
> **Purpose**: Configure agent capabilities on a class.

## Signature

```python
def agentic_object(
    imports: list[object] = [],
    invoke_sub_agents: bool = False,
    allow_code_execution: bool = False,
) -> Callable[[type], type]:
    """Configure agent capabilities for an AgenticObject subclass."""
    ...
```

## Description

Class-level decorator that controls what capabilities the agent has when working on instances of this class. All three parameters default to `False` or empty — every capability is opt-in.

## Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `imports` | `list[module]` | `[]` | Modules injected into the sandbox as Python object references |
| `invoke_sub_agents` | `bool` | `False` | Enables `invoke()` for sub-agent calls on this object |
| `allow_code_execution` | `bool` | `False` | Allows the agent to write and execute sandboxed Python code when invoked on this class |

## Usage

```python
from dataclasses import dataclass
from peteos import AgenticObject, tool, agentic_object
import time
import decimal

# Only sandboxed code with time and decimal modules
@agentic_object(imports=[time, decimal], allow_code_execution=True)
class TimeRecord(AgenticObject):
    ...

# Only sub-agent invocation enabled
@agentic_object(invoke_sub_agents=True)
class Child(AgenticObject):
    ...

# All three capabilities
@agentic_object(imports=[time, decimal], invoke_sub_agents=True, allow_code_execution=True)
class PriceRecord(AgenticObject):
    ...

# No capabilities beyond @tool methods (default)
class InventoryItem(AgenticObject):
    ...
```

## Parameter Details

### `imports`

Modules passed to the sandbox as Python object references. The agent never sees them as import strings — they're injected directly into sandbox globals.

```python
@agentic_object(imports=[time, decimal])
class PriceRecord(AgenticObject):
    # Agent's sandboxed code can use:
    #   time.time()
    #   decimal.Decimal("100.00")
```

- Modules are stored in a private registry keyed by class object
- The agent cannot discover another class's imports — the attribute does not exist on instances
- Inheritance does not merge imports; each class declares its own

### `invoke_sub_agents`

Enables the `invoke()` member function on this object, allowing other agents to invoke sub-agents on instances of this class.

- The gatekeeper checks this flag on the **target** object
- `invoke()` is available in sandboxed code when `allow_code_execution=True` on the calling object
- Only the target needs this flag — the calling agent's class does not need it
- Setting this to `True` on every class enables uncontrolled recursion; set selectively

### `allow_code_execution`

When `True`, the agent invoked on this object is allowed to write and execute sandboxed Python code. It does not control object access — the sandbox always has `self` as the root object.

The agent can use sandboxed code to:
- Create new agentic objects and append them to collections
- Loop over list/dict members
- Perform calculations
- Call methods (decorated or not) on reachable objects

- The sandbox provides exactly one global variable: `self` (the root object instance)
- No `__builtins__`, no `__import__`, no network, no file system access
- Modules from `imports` are injected as additional globals

## Security Notes

- Imports stored in a private registry — not accessible via class attributes
- Attempting `sub_obj.__class__.agent_allowed_imports` raises `AttributeError`
- Agent cannot enumerate sandbox globals or discover modules it wasn't given
- `__builtins__` is set to an empty dict in the sandbox
- `__import__` and `importlib` are not available in any form

## Members

### `acquire(timeout: float | None = None) -> None`

Acquires the `threading.Lock` that serializes invocations on this object. Only one invocation can hold the lock at a time — both `invoke_agent()` and `AgenticObject.invoke()` call `acquire()` before creating a session and `release()` when done.

This prevents race conditions caused by interleaved `@tool` calls that read/write the object's member variables.

| Scenario | Behavior |
|---|---|
| Lock available | Acquires immediately, returns `None` |
| Lock held by another caller | Blocks until the lock is released |
| Lock not acquired within `timeout` seconds | Raises `TimeoutError` |
| `timeout` is `None` | Blocks indefinitely |

**Concurrency Invariant:** The lock is acquired on the **target object**, not on a `thread_id`. Two concurrent calls to the same object — same `thread_id` or different — both block. The object is the protection boundary.

### `release() -> None`

Releases the `threading.Lock` acquired by a prior `acquire()` call. Signals that the invocation is complete and other callers waiting on the same object can proceed.

| Scenario | Behavior |
|---|---|
| Lock held by caller | Releases the lock, unblocks waiting callers |
| Lock not held | Raises `RuntimeError` (double-release protection) |

`invoke_agent()` and `AgenticObject.invoke()` handle `release()` via try/finally to guarantee it's always called even on exceptions.

## Related

- [`invoke_agent()`](invoke_agent.md) — main entry point for agent-driven object interaction
- [`AgenticObject.invoke()`](invoke.md) — sub-agent invocation from sandboxed code
