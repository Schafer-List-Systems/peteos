# Configuration

How agentic objects connect to LLMs and how user overrides customize their behavior.

## Overview

An agentic object's configuration flows through two layers:

1. **Backends** — which LLM API the agent talks to (`ChatBotManager`)
2. **Roles** — the agent's identity, model selection, and system prompt hooks (`Role`, `AgenticObjectRegistry`, `RoleManager`)

These layers work together. The backend provides the models. The role picks which model to use.

## Pages

| Page | Description |
|---|---|
| [Backends](./backends.md) | Register LLM backends and load configuration from JSON |
| [Roles](./roles.md) | Build canonical roles from agentic classes, merge user overrides, and control instance isolation |
