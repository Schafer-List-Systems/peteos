# Agentic Object Roles

Every agentic object implicitly has its own role.
A role defines the identity and behavior of an agentic object — its system prompt, its toolset, and which model it uses to reason.

See [Role reference](../reference/role.md) for the full field list.

## Role Naming

Each agentic object class gets a role name automatically.
By default the name is `oap_{ClassName}`.
You can override this via the `@agentic_object(role="...")` decorator:

```python
@agentic_object(role="finance_analyst")
class FinancialReport(AgenticObject):
    """You are a financial analyst."""
```

This is useful when the same class name appears in different modules or packages and you want to avoid naming conflicts.

## Creating a Role

When you instantiate an agentic object, the role is built in two steps:

1. **Canonical role** — the name, system prompt, model, and tool filter are built from the class's docstrings and `@agentic_object` configuration.
The system prompt is built by concatenating docstrings of all agentic parent classes in the inheritance order.

2. **User overrides** — any roles registered in `RoleManager` (manually or loaded from disk) are merged into the canonical role.
See [RoleManager reference](../reference/rolemanager.md) and [Role configuration](../config/roles.md) for how to do this.
The user can override `model` and `description`, and append to `system_prompt` and `system_prompt_hooks`.
All other fields — `required_tools`, `auto_approve_tools`, `tool_filter`, `execution_environment`, `behavior_policy` — remain canonical and are not overridden.

Each instance receives an independent copy of the merged role, so one instance never affects another.

## Model Selection and Cost Control

The role's `model` field is a regex pattern that determines which available model from a backend the agent will use.
This gives you fine-grained control over which model family each agentic object class uses — for example, applying an expensive model where it is needed and a cheaper one elsewhere.
See [Backends configuration](../config/backends.md) for how to set up the available models.

```python
# Expensive model for complex reasoning
RoleManager.register_role(Role(
    name="oap_FinancialReport",
    model="claude-3-opus-.*",
))

# Cheaper model for simple tasks
RoleManager.register_role(Role(
    name="oap_ItemList",
    model="gpt-4o-mini.*",
))
```

## Examples

### Configuring via manual registration

```python
from peteos.persona.role import Role
from peteos.persona.rolemanager import RoleManager

RoleManager.register_role(Role(
    name="oap_SupportBot",
    model="gpt-4o-mini.*",
    description="A lightweight customer support agent",
))

bot = SupportBot()  # Uses gpt-4o-mini
```

### Configuring via disk loading

```python
RoleManager.load_from_dir("/config/roles/")
```

The directory structure:

```
oap_SupportBot/
  description.md        # A lightweight customer support agent
  system_prompt.md      # You handle common support requests...
  config.json           # {"model": "gpt-4o-mini.*"}
```

### Overriding the role name

```python
@agentic_object(role="email_analyst")
class ReportAnalyzer(AgenticObject):
    """You analyze emails."""

RoleManager.register_role(Role(
    name="email_analyst",
    model="claude-3-opus-.*",
))
```
