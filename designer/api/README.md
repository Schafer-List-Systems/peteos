# API Documentation

## Structure

```
api/
├── README.md                ← Master index
├── getting-started.md       ← Tutorial: first agentic object
├── examples/                ← Concrete examples
│   ├── data-cleansing/README.md
│   ├── sub-object-invocation/README.md
│   ├── simulation-decision-making/README.md
│   └── dynamic-object-creation/README.md
└── reference/               ← API reference
    ├── tool.md              ← @tool decorator
    ├── agentic_object.md      ← @agentic_object decorator
    ├── invoke_agent.md      ← invoke_agent() function
    ├── invoke.md            ← AgenticObject.invoke() method
    └── error.md             ← Error class
```

## Files

| File | Purpose |
|---|---|
| [getting-started.md](getting-started.md) | Tutorial: walk through creating your first agentic object |
| [examples/data-cleansing.md](examples/data-cleansing.md) | 4 concrete examples with expected output: data cleansing, sub-object invocation, simulation decision-making, dynamic object creation |
| [reference/tool.md](reference/tool.md) | `@tool` decorator — marks methods as agent tools |
| [reference/agentic_object.md](reference/agentic_object.md) | `@agentic_object` decorator — configures agent capabilities per-class |
| [reference/invoke_agent.md](reference/invoke_agent.md) | `invoke_agent()` function — main entry point for agent-driven object interaction |
| [reference/invoke.md](reference/invoke.md) | `AgenticObject.invoke()` method — invoke sub-agents on child objects |
| [reference/error.md](reference/error.md) | `Error` class — task failure result, returned not raised |
