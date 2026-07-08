# Invocation Hooks

Hooks let you observe and control agent invocations. Pass a `hooks` dict as the `hooks` parameter to [`invoke_agent()`](../reference/agentic-object-base.md#invoke_agent):

```python
hooks = {
    "on_invoke": [lambda ctx: print(ctx["prompt"])],
    "on_invoke_complete": [lambda ctx: print(ctx["result"])],
}
result = await obj.invoke_agent(
    prompt="Analyze this data.",
    hooks=hooks,
)
```

## Hook Types

### `on_invoke`

Fires **before** the agent starts reasoning. You can inspect the prompt or prevent the invocation.

```python
def log_and_guard(ctx):
    print(f"[{ctx['role']}] Prompt: {ctx['prompt']!r}")
    return None  # allow
    # return "Error message"  # abort and return to caller

result = await obj.invoke_agent(
    prompt="Analyze this data.",
    hooks={"on_invoke": [log_and_guard]},
)
```

The hook receives a context dict with `role`, `prompt`, and `session`. Return `None` to allow the invocation, or a non-`None` string to abort it — the agent is not started and the string is returned as an `Error` to the caller.

### `on_invoke_complete`

Fires **after** the agent finishes, regardless of outcome.

```python
def log_result(ctx):
    print(f"[{ctx['role']}] Done: {type(ctx['result']).__name__}")

result = await obj.invoke_agent(
    prompt="Analyze this data.",
    hooks={"on_invoke_complete": [log_result]},
)
```

The context dict includes `result` — the agent's return value or an `Error` object.

### `on_tool_call`

Fires **before each tool execution**. Use this to monitor or block specific tools.

```python
def block_python_exec(ctx):
    if ctx["tool_name"] == "python_exec":
        return "Code execution is not allowed"
    return None  # allow

result = await obj.invoke_agent(
    prompt="Write a file and count its lines.",
    hooks={"on_tool_call": [block_python_exec]},
)
```

The context dict includes `role`, `session`, `tool_name`, and `arguments`.
Return `None` to allow execution, or a non-`None` string to deny it — the tool and all remaining tools in the same group are skipped, and the agent is given another reasoning turn.

## Recursive Propagation

Hooks are automatically forwarded to every sub-agent invocation.
If an agentic object calls another via `invoke()` or `python_exec`, the same hooks fire for the sub-agent, giving you visibility across the entire invocation tree.

## Combining Hooks

Register multiple hooks per list.
They execute in registration order.

```python
def log_before(ctx):
    print(f"[{ctx['role']}] starting")

def log_after(ctx):
    print(f"[{ctx['role']}] done: {type(ctx['result']).__name__}")

result = await obj.invoke_agent(
    prompt="Analyze this data.",
    hooks={
        "on_invoke": [log_before],
        "on_invoke_complete": [log_after],
    },
)
```