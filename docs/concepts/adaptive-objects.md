# Adaptive Objects

Adaptive objects are agentic objects that can manipulate and extend their own code at runtime. They add new source code or remove previously added code — dynamically reshaping their capabilities while operating.

## The Adaptive Object

An agentic object is an object with well-defined state and behavior. An adaptive object extends this by being able to add or remove functions during execution. These changes affect its toolset immediately, so the agentic object can reason about and call the new functions in subsequent invocations.

## Three Levels of Adaptation

Adaptive objects can manipulate on three levels, each with different visibility:

### Class Level

Changes on the class level are immediately visible to every agentic object in every session that belongs to this class. Adding a function at this level extends the capabilities for all instances and all ongoing conversations.

### Instance Level

Changes on the instance level are only visible to the particular agentic object instance that made them, but across every session or conversation belonging to that instance. One instance's adaptations do not affect other instances of the same class.

### Session Level

Changes on the session level are only visible within that particular session. Other sessions and invocations are unaffected.

## Example

The [full example](../examples/fibonacci-squared-2.py) contains the complete runnable code. The core idea looks like this:

```python
@agentic_object(allow_code_execution=True)
class FibonacciSquared(AdaptiveObject):
    """You are an efficient calculator for custom functions."""

sq = FibonacciSquared()
for n in [10, 15, 20]:
    prompt = f"Compute the sequence where each element is the sum of the squares of its two predecessors. Start with 0, 1. Compute the {n}-th element."
    print(await sq.invoke_agent(prompt, output_schema=int))
```

On the first invocation the agentic object reasons about how to compute the sequence, builds a helper function, tests it, and registers it as a tool. On the second and third invocations it already has the function and simply calls it — making subsequent calls much faster.

## Test-Driven Extension

When an adaptive object extends its functionality by adding new source code, it is encouraged to provide unit tests for those functions. These tests are executed in an isolated sandbox before the new code is registered. The function is only added if all tests pass successfully, ensuring that extensions are reliable before they affect the object's behavior.

## Current and Future Capabilities

The current implementation allows agentic objects to adapt at the instance level. A known limitation is that agentic objects can only adapt on the instance level — class-level and session-level adaptation are not yet supported.

Future capabilities include:

- **Code quality improvements** — automated checks to verify the quality of agent-generated code.
- **Persistent code across restarts** — saving agent-defined functions so they survive application restarts.
- **Broader adaptation levels** — enabling class-level and session-level adaptation.

## Related Concepts

- [Agentic Object Roles](./agentic-object-roles.md) — how roles define the identity and toolset of agentic objects
- [Composition](./composition.md) — combining multiple agentic classes
- [Testing](./testing.md) — testing patterns in the sOAP paradigm
