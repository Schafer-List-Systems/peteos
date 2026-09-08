# PeteOS

Simple Object-Agentic Programming (sOAP) — an agentic application framework that brings together object-oriented programming and AI agents.
The easiest way to build reliable and powerful multi-agent systems.

Peteos lets you build **agentic objects**: ordinary Python objects that can think, decide, and act on their own.
The agent can access and modify the object's own state through its tools, enabling self‑aware behavior.
No external memory stores or bolt-on intelligence — the object itself is the center of persistence and agency.

## Quick Start

After installing and configuring Peteos, you can build and use agentic objects like the following. 

```python
import asyncio
from peteos import AgenticObject, tool

class Example(AgenticObject):
    """You are Pete, a concise assistant."""
    def __init__(self, job: str):
        super().__init__()
        self._job = job
    
    @tool
    def get_job(self) -> str:
        return self._job

async def main():
    pete = Example("demonstrator")
    result = await pete.invoke_agent("Hello, what's your name and job?", output_schema=list[str])
    print(result)  # ['Pete', 'demonstrator']

asyncio.run(main())
```
This example shows how an agentic object can reason about a query and use its own tool to access its internal state, combining AI reasoning with ordinary object behavior.

## Key Features

The documentation and the examples cover more complex agentic objects with the following key features. 

- **Agentic objects** — derive from `AgenticObject` and get a thinking agent behind every instance.
- **Inheritance** — inherit from agentic objects to refine the system prompt and extend its toolset.
- **Multi-inheritance composition** — combine agentic classes to compose behavior.
- **Structured output** — declare `output_schema` and get typed results back.
- **Sandboxed code execution** — let agents run Python in a restricted sandbox.
- **Persistent sessions** — threads, memory, and sub-agent coordination via `persistent_thread_id`.
- **Test-driven development** — it's OOP, so test your agents as you test any other software.
- **Debuggable agents** — it's plain Python, so debug your agents with a standard debugger.


## Installation

Install PeteOS via PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install peteos
```

## Configuration

Copy `peteos.json.example` to `peteos.json` and adapt it to your environment:

```bash
cp peteos.json.example peteos.json
```

A small example for the `peteos.json` when running [ollama](https://ollama.com/download) locally with [glm-4.7](https://ollama.com/library/glm-4.7-flash:latest) installed:
```
{
  "backends": [
    {
      "name": "ollama",
      "url": "http://localhost:11434",
      "model_priorities": {
        "glm-4.7-flash:latest": 100
      },
      // maximum number of output tokens per LLM-API Call
      "max_output": 16384
    }
  ]
}
```

The configuration file defines chatbot backends (LLM providers) that Peteos uses for agent reasoning:

- **name** — a descriptive label for the backend
- **url** — the API endpoint for the provider
- **api_type** — `anthropic`, `gemini`, or omitted for OpenAI-compatible endpoints
- **api_key** — environment variable name holding the API key (never hardcode keys)
- **model_priorities** — maps model names to priority scores (higher = preferred)

To verify that your configuration is functional run
```
python3 examples/00_hello_pete.py
```


## Resources

- [Getting Started](https://github.com/Schafer-List-Systems/peteos_docs/blob/0.3.x/getting-started.md) — installation and first example
- [Introduction](https://github.com/Schafer-List-Systems/peteos_docs/blob/0.3.x/introduction.md) — the sOAP paradigm
- [Concepts](https://github.com/Schafer-List-Systems/peteos_docs/blob/0.3.x/concepts/index.md) — composition, invocation, state, testing
- [Reference](https://github.com/Schafer-List-Systems/peteos_docs/blob/0.3.x/reference/index.md) — API docs, agentic objects
- [Best Practices](https://github.com/Schafer-List-Systems/peteos_docs/blob/0.3.x/best-practices/index.md) — Best practices when using PeteOS
- [Examples](https://github.com/Schafer-List-Systems/peteos_docs/tree/0.3.x/examples) — Examples show-casing some features

## Licensing

This project is dual-licensed:

- **Non-commercial use** (including evaluation of commercial use cases) is free
  under the license in the [LICENSE](LICENSE) file.
- **Commercial use** (production, products, services, SaaS, etc.) requires a
  commercial license from the Schäfer List Systems GmbH.

If you intend to use this software commercially, please contact us at
<info@schaeferlist.de> or visit <https://www.schaeferlist.de>.
