# Getting Started

## The Idea

Object-agentic programming lets you write normal Python classes and expose their parts to an agentic AI. Mark methods with `@tool`. Call `invoke_agent`. The agent does the rest.

## Step 1: Create an Agent Object

```python
from peteos import AgenticObjectBase, tool

class InventoryManager(AgenticObjectBase):
    def __init__(self):
        self._items = ["Flour", "Hammer", "Sugar", "Nails", "Paint"]

    @tool
    def get_items(self) -> list[str]:
        """Return the current items."""
        return self._items

    @tool
    def set_items(self, items: list[str]) -> None:
        """Replace the items list."""
        self._items = items

    def internal_method(self):
        """Not decorated — not a direct tool call."""
        ...
```

The agent has `get_items` and `set_items` as direct tool calls. `internal_method` is not available as a tool call.

Key points:
- **Inherit** from `AgenticObjectBase`
- **`@tool`** on every method you want the agent to call directly — getters, setters, and actions
- Read-only: only a getter method (no setter)
- Read-write: getter + setter, both decorated with `@tool`
- Undecorated methods are not tool calls

## Step 2: Invoke an Agent

Invoke an agent on an object. The agent sees the decorated methods, reasons about them, and works on the object.
It returns a structured result matching `output_schema` — or an `Error` object if it could not produce the desired result.

```python
from dataclasses import dataclass
from peteos import invoke_agent

@dataclass
class Result:
    items: list[str]

result = invoke_agent(
    InventoryManager(),
    prompt="List all groceries in the inventory.",
    output_schema=Result,
)
```

**Expected output:**

```python
Result(items=["Flour", "Sugar"])
```

Only "Flour" and "Sugar" are groceries — "Hammer", "Nails", and "Paint" are not.
Classifying items as groceries requires semantic understanding, not a simple filter.
This example shows the benefit of having an agent as a first-class consumer of an object's interface.

- **`prompt`** — what to tell the agent
- **`output_schema`** — expected return type (dataclass, etc.). Omitted for `str`
- The agent discovers all `@tool` methods on the object automatically
- Returns either a `Result` instance or an `Error` object (not an exception)

## Step 3: Threads

By default, `invoke_agent` creates a non-persistent session — the agent forgets all context as soon as the call finishes.
Providing a `thread_id` creates a persistent session that carries forward the conversation history.

**Non-persistent (default):** the agent starts fresh on every call.

```python
result1 = invoke_agent(obj, prompt="Remember: my favorite color is blue.")
result2 = invoke_agent(obj, prompt="What is my favorite color?")
# result2: Error("I don't have that information.")
```

The agent has no memory of the previous call.

**Persistent:** the agent retains context across calls with the same `thread_id`.

```python
result1 = invoke_agent(obj, prompt="Remember: my favorite color is blue.", thread_id="abc123")
result2 = invoke_agent(obj, prompt="What is my favorite color?", thread_id="abc123")
# result2: Result(items=["blue"])
```

The agent remembers from the prior call in the same thread.

## Step 4: Code Execution

The agent can write sandboxed Python code to iterate, create objects, call methods, and perform calculations — but only if allowed.

```python
@agentic_object(imports=[time, decimal], allow_code_execution=True)
class PriceRecord(AgenticObjectBase):
    ...
```

The `allow_code_execution` flag (default `False`) controls whether the agent can write sandboxed code on this object. `@agentic_object(imports=[...])` declares which modules are available in the sandbox. The agent never sees them as import strings — they're injected as real Python object references.

**Expected output:** The agent writes sandboxed code like `self.items.append(PriceRecord(raw_value="100 USD"))` and executes it. The object is mutated directly — `items` now contains a new `PriceRecord`. No LLM messages are involved — just direct Python execution in the sandbox.

## What Happens

1. `invoke_agent` reflects on the object, collecting all `@tool` decorated methods
2. It builds a schema of available tools
3. It runs the agent with the prompt and the object's interface
4. The agent reasons, calls tools, reads/writes state via getters/setters, executes sandboxed code
5. It returns structured output (matching `output_schema`) or an `Error` object
6. Exceptions are raised only for API failures, not task failures

## What You Get

- Structured data from the agent — not free text
- Agent can mutate the object — getters and setters provide full control
- Agent can reason over sub-objects with context isolation
- Agent can invoke sub-agents on nested agentic objects
- Agent can create new agentic objects and add them to collections via sandboxed code
- The developer controls everything via decorators

## Step 5: Sub-Objects

Objects can contain other agentic objects as member variables. The parent agent invokes sub-agents on children via sandboxed code, reasoning about their state independently.

```python
from dataclasses import dataclass
from peteos import AgenticObjectBase, tool, agentic_object
from peteos import invoke_agent

@dataclass
class Answer:
    response: str

@agentic_object(invoke_sub_agents=True)
class Child(AgenticObjectBase):
    def __init__(self, name: str, hunger: float, tiredness: float):
        self._name = name
        self._hunger = hunger  # 0-100
        self._tiredness = tiredness  # 0-100

    @tool
    def get_state(self) -> dict:
        """Return current state."""
        return {"name": self._name, "hunger": self._hunger, "tiredness": self._tiredness}

@agentic_object(allow_code_execution=True)
class Parent(AgenticObjectBase):
    def __init__(self):
        self._children = [
            Child("Alice", hunger=70.0, tiredness=30.0),
            Child("Bob", hunger=20.0, tiredness=80.0),
        ]

    def get_children(self) -> list[Child]:
        """Return list of children."""
        return self._children

    def prepare_food(self) -> str:
        """Prepare food for the children (not a tool call)."""
        return "Food is ready."

    def play_with_children(self) -> str:
        """Take all children out to play (not a tool call)."""
        return "Played with children."

    def bring_to_bed(self, child: Child) -> str:
        """Bring a child to bed (not a tool call)."""
        return f"Brought {child._name} to bed."
```

The agent asks each child via sub-agent invocation, then acts via sandboxed code:

```python
invoke_agent(
    Parent(),
    prompt="Ask each child if they want to play using sandboxed code (self.invoke(child, 'Do you want to play?')). "
           "If a child is too hungry or too tired, bring them to bed via sandboxed code. "
           "Otherwise, play with all willing children via sandboxed code. "
           "If any child was hungry, prepare food via sandboxed code.",
    thread_id="parent-tick",
)
```

**Expected output:** The agent invokes a sub-agent on each child to ask if they want to play. Children that are too tired get brought to bed. The hungry child causes food to be prepared. The actions are executed via sandboxed code.

The `@agentic_object(invoke_sub_agents=True)` decorator on the child enables:
- The parent agent to invoke sub-agents on this child via `self.invoke(child, "Do you want to play?", output_schema=Answer)`
- Each child to reason about its own state in an isolated context
