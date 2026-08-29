# Role

A Role defines an agent's identity and behavior: its name, the model it uses, and how it interacts with tools.

## Constructor

```python
from peteos.persona.role import Role

Role(
    name: str,
    description: str | None,
    system_prompt: str | None = None,
    system_prompt_hooks: list[Callable[[], str]] | None = None,
    required_tools: list[str] | None = None,
    execution_environment: str = "REPL",
    model: str = ".*",
    auto_approve_tools: list[str] | None = None,
    tool_filter: list[str] | None = None,
    behavior_policy: str = "responsive",
    max_truncation_retries: int = 2,
    max_output_turns: int = 3,
    max_output_attempts: int = 3,
)
```

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Unique identifier for the role. |
| `description` | `None` | Brief description of what the agent does. |
| `system_prompt` | `None` | Static text that forms the agent's initial instructions. |
| `system_prompt_hooks` | `None` | Callbacks invoked at session creation to append dynamic fragments to the system prompt. |
| `required_tools` | `None` | Tools that must be available for this role. |
| `execution_environment` | `"REPL"` | Execution environment identifier. |
| `model` | `".*"` | Regex pattern to match model IDs. |
| `auto_approve_tools` | `None` | Tool names the agent can call without user approval. |
| `tool_filter` | `None` | Regex patterns. Only tools whose names match any pattern are visible to the agent. |
| `behavior_policy` | `"responsive"` | Agent behavior strategy. `"responsive"` yields on text output; `"continuous"` keeps looping until yield. |
| `max_truncation_retries` | `2` | Maximum number of retries due to token window truncation. |
| `max_output_turns` | `3` | Maximum number of output-producing turns per `invoke_agent` call before giving up. |
| `max_output_attempts` | `3` | Maximum number of `produce_output` call attempts per output turn before failing. |

## Methods

| Method | Description |
|---|---|
| `add_system_prompt_hook(hook) → None` | Add a callback that returns a dynamic fragment for the system prompt. |
| `load_from_dict(data) → Role` | Create a new Role from a dictionary. |
| `load_from_path(path) → Role` | Create a new Role from a directory containing `description.md` and optional files. |
