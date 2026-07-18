# Peteos

Simple Object-Agentic Programming (sOAP) — an agentic application framework that brings together object-oriented programming and AI agents.

Peteos lets you build **agentic objects**: ordinary Python objects that can think, decide, and act on their own. No external memory stores or bolt-on intelligence — the object itself is the center of persistence and agency.

## Quick Start

```python
class Example(AgenticObject):
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

### From a distribution package (recommended)

Download a `.tar.gz` archive and extract it, then install the wheel inside:

```bash
tar xzf peteos-0.0.1.tar.gz
cd peteos-0.0.1
python -m venv .venv
source .venv/bin/activate
pip install peteos-0.0.1-cp312-cp312-linux_x86_64.whl
```

### From a Python package (PyPI / direct install)

```bash
python -m venv .venv
source .venv/bin/activate
pip install peteos
```

### From source (development)

```bash
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/yourusername/peteos.git
cd peteos
pip install -e ".[dev]"
```

> **Note:** This installs from source code. Cython extensions are compiled in-place via `python setup.py build_ext --inplace`. See the [development guide](docs/developer/development.md) for build options.

## Configuration

Copy `peteos.json.example` to `peteos.json` and adapt it to your environment:

```bash
cp peteos.json.example peteos.json
```

The configuration file defines chatbot backends (LLM providers) that Peteos uses for agent reasoning:

- **name** — a descriptive label for the backend
- **url** — the API endpoint for the provider
- **api_type** — `anthropic`, `gemini`, or omitted for OpenAI-compatible endpoints
- **api_key** — environment variable name holding the API key (never hardcode keys)
- **model_priorities** — maps model names to priority scores (higher = preferred)

## Key Features

- **Agentic objects** — derive from `AgenticObject` and get a thinking agent behind every instance.
- **Multi-inheritance composition** — combine agentic classes with `AgenticObject` to compose behavior.
- **Structured output** — declare `output_schema` and get typed results back.
- **Sandboxed code execution** — let agents run Python in a restricted sandbox.
- **Persistent sessions** — threads, memory, and sub-agent coordination via `persistent_thread_id`.
- **Chatbot backends** — configure OpenAI and Anthropic-compatible local AI servers via a singleton manager.
- **Test-driven development** — OOP-aligned unit but Monte Carlo iterations for non-deterministic behavior.

## Resources

- [Installation Guide](docs/developer/development.md) — developer setup and build instructions
- [Introduction](docs/introduction.md) — the sOAP paradigm
- [Getting Started](docs/getting-started.md) — installation and first example
- [Concepts](docs/concepts/index.md) — composition, invocation, state, testing
- [Reference](docs/reference/index.md) — API docs, agentic objects
- [Best Practices](docs/best-practices/index.md) — Best practices when using PeteOS
- [Examples](docs/examples/) — Examples show-casing some features
