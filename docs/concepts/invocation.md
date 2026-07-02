# Invocation

An agentic object is invoked through the `invoke_agent()` method. The agent reasons about your prompt and responds, using tools to interact with the object's state as needed.

## Basic Invocation

Invoke an agentic object directly with a prompt:

```python
pete = HelloPete()
result = await pete.invoke_agent("Hello, what's your name?")
print(result)  # My name is Pete.
```

### Input Modality

A prompt is always text. An optional `image` parameter accepts a local file path or HTTP(S) URL to attach an image to the prompt. The image is base64-encoded and sent alongside the text prompt.

### Output Schema

The `output_schema` parameter forces the agent to return structured output matching the provided type. This enables converting unstructured information from an agent into structured data.

Supported schema types:
- Scalar primitives: `str`, `int`, `float`, `bool`
- Collection types: `list`, `list[T]`, `dict`, `dict[K, V]`
- `Enum` with string values
- `dataclass` (including nested dataclasses, self-referential structures, and arbitrary nesting depth)
- Union types (e.g. `str | int`, `MyClass | None`)

When JSON parsing fails and the schema is a scalar or enum, the raw string is interpreted directly as the value. This handles cases where the agent returns a bare string instead of a JSON-encoded one.

```python
from dataclasses import dataclass

@dataclass
class UserProfile:
    name: str
    age: int
    active: bool

result = await pete.invoke_agent(
    "Based on this conversation, extract the user's profile.",
    output_schema=UserProfile,
)
print(result)  # UserProfile(name="Alice", age=30, active=True)
```

## Thread ID and Persistence

Invocations can be transient or persistent, controlled by the `persistent_thread_id` parameter.

### Persistent Sessions

Providing a `persistent_thread_id` persists the session so you can continue a conversation across multiple invocations. This gives the agent short-term memory about the recent conversation.

```python
result = await obj.invoke_agent("Remember this: the answer is 42.", persistent_thread_id="my-session")
result = await obj.invoke_agent("What number did I ask you to remember?", persistent_thread_id="my-session")
```

The `persistent_thread_id` is an arbitrary string you choose. Subsequent invocations with the same thread ID continue the same conversation.

### Transient Sessions

When no `persistent_thread_id` is provided, the conversation is anonymous and temporary. The session and runner are destroyed after the invocation completes, and every subsequent call starts with a fresh context.

```python
result = await obj.invoke_agent("This conversation will not be remembered.")
```

## Sub-Agent Invocation

Agentic objects can invoke sub-agents in two ways:

### Via Sandbox

When sandboxed code execution is enabled on an agentic object, the sandbox provides a `python_exec` tool that the agent can call. Within the sandbox, the Python code accesses the agentic object through the `self` parameter of the `func(self)` signature.

### Via `invoke()`

When `invoke_sub_agents` is enabled on the agentic class, the agent can invoke another agentic object's agent through the `invoke()` method:

```python
from peteos.oap.decorators import agentic_object

@agentic_object(invoke_sub_agents=True)
class Supervisor(AgenticObject):
    """You are a supervisor. Delegate tasks to specialists."""

    def __init__(self):
        super().__init__()
        self.specialist = Specialist()

result = await supervisor.invoke(
    target=supervisor.specialist,
    prompt="Analyze this data and classify it.",
    persistent=True,
)
```

The `persistent` flag controls whether the parent's thread ID is forwarded. When `persistent=True`, sub-agent calls inherit the same thread ID, allowing continuity. When `persistent=False`, the sub-agent call is transient.

### Thread ID Forwarding

When an agentic object invokes sub-agents (via sandboxed code), thread IDs are forwarded. If a parent invocation has a `persistent_thread_id`, sub-agent calls inherit the same thread ID, allowing the agent to maintain continuity when delegating work to other objects or itself.
