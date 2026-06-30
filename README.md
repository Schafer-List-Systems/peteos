# Peteos

Simple Object-Agentic Programming (sOAP) — an agentic application framework that brings together object-oriented programming and AI agents.

Peteos lets you build **agentic objects**: ordinary Python objects that can think, decide, and act on their own. No external memory stores or bolt-on intelligence — the object itself is the center of persistence and agency.

## Quick Start

```python
class Example(AgenticObjectBase):
    """You are Pete, a concise assistant."""
    def __init__(self, job: str):
        super().__init__()
        self._job = job
    
    @tool
    def get_job(self) -> str:
        return self._job

pete = Example("demonstrator")
result = await pete.invoke_agent("Hello, what's your name and job?", output_schema=list[str])
print(result)  # ['Pete', 'demonstrator']
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/yourusername/peteos.git
cd peteos
pip install .
```

## Key Features

- **Agentic objects** — derive from `AgenticObjectBase` and get a thinking agent behind every instance.
- **Multi-inheritance composition** — combine agentic classes with `AgenticObjectBase` to compose behavior.
- **Structured output** — declare `output_schema` and get typed results back.
- **Sandboxed code execution** — let agents run Python in a restricted sandbox.
- **Persistent sessions** — threads, memory, and sub-agent coordination via `persistent_thread_id`.
- **Chatbot backends** — configure OpenAI and Anthropic-compatible local AI servers via a singleton manager.
- **Test-driven development** — OOP-aligned unit but Monte Carlo iterations for non-deterministic behavior.

## Resources

- [Introduction](docs/introduction.md) — the sOAP paradigm
- [Getting Started](docs/getting-started.md) — installation and first example
- [Concepts](docs/concepts/) — composition, invocation, state, testing
- [Reference](docs/reference/) — API docs, agentic objects
