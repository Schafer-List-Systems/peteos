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
    """Configure agent capabilities for an AgenticObjectBase subclass."""
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
from peteos import AgenticObjectBase, tool, agentic_object
import time
import decimal

# Only sandboxed code with time and decimal modules
@agentic_object(imports=[time, decimal], allow_code_execution=True)
class TimeRecord(AgenticObjectBase):
    ...

# Only sub-agent invocation enabled
@agentic_object(invoke_sub_agents=True)
class Child(AgenticObjectBase):
    ...

# All three capabilities
@agentic_object(imports=[time, decimal], invoke_sub_agents=True, allow_code_execution=True)
class PriceRecord(AgenticObjectBase):
    ...

# No capabilities beyond @tool methods (default)
class InventoryItem(AgenticObjectBase):
    ...
```

## Parameter Details

### `imports`

Modules passed to the sandbox as Python object references. The agent never sees them as import strings — they're injected directly into sandbox globals.

```python
@agentic_object(imports=[time, decimal])
class PriceRecord(AgenticObjectBase):
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
