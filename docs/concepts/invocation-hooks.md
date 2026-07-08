# Invocation Hooks

Hooks let you observe and control agent invocations.
Pass a `hooks` dict to [`invoke_agent()`](../reference/agentic-object-base.md#invoke_agent):

```python
hooks = {
    "on_invoke": [lambda ctx: print(ctx["prompt"])],
    "on_invoke_complete": [lambda ctx: print(ctx["result"])],
}
result = await obj.invoke_agent(prompt, hooks=hooks)
```

## Hook Types

### `on_invoke`

Fires **before** the agent starts reasoning.

```python
hooks = {
    "on_invoke": [
        lambda ctx: print(f"Calling {ctx['role']}..."),
        lambda ctx: None if ctx["prompt"] else "Prompt is empty — skipping",
    ],
}
```

The hook receives a context dict with `role`, `prompt`, and `session`.
Return `None` to allow the invocation, or a non-`None` string to abort it — the agent is not started and the string is returned as an `Error` to the caller.

### `on_invoke_complete`

Fires **after** the agent finishes, regardless of outcome.

```python
hooks = {
    "on_invoke_complete": [
        lambda ctx: print(f"{ctx['role']} finished: {ctx['result']!r}"),
    ],
}
```

The context dict includes `result` — the agent's return value or an `Error` object.

### `on_tool_call`

Fires **before each tool execution**.
Use this to monitor or block specific tools.

```python
hooks = {
    "on_tool_call": [
        lambda ctx: None if ctx["tool_name"] != "python_exec" else "Code execution is not allowed",
    ],
}
```

The context dict includes `role`, `session`, `tool_name`, and `arguments`.
Return `None` to allow execution, or a non-`None` string to deny it — the string is sent to the agent as a tool result error, allowing it to reason about the denial.

## Recursive Propagation

Hooks are automatically forwarded to every sub-agent invocation.
If an agentic object calls another via `invoke()` or `python_exec`, the same hooks fire for the sub-agent, giving you visibility across the entire invocation tree.

## Combining Hooks

Multiple hooks run in registration order within each list.

```python
def log_before(ctx):
    print(f"[{ctx['role']}] starting")

def log_after(ctx):
    print(f"[{ctx['role']}] done: {type(ctx['result']).__name__}")

hooks = {
    "on_invoke": [log_before],
    "on_invoke_complete": [log_after],
}
```
