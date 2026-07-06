# sOAP Philosophy

sOAP is built on a three-layer model that extends object-oriented programming with agency and persistence.

## The Three Layers

| Layer | Analogy | Role |
|---|---|---|
| **Agentic Class** | OOP class | Blueprint for agentic behavior — system prompt, tools, configuration |
| **Agentic Object** | OOP object instance | A living instance of the class with its own internal state |
| **Session** | Persistent conversation thread | An independent conversation context tied to a single agentic object |

This is OOP extended to three levels rather than the usual two. Agentic classes and agentic objects are regular OOP classes and objects — references, inheritance, and composition work as you expect.

### Agentic Objects

Agentic objects are created by instantiating an agentic class. A single agentic class can produce many agentic object instances. Each agentic object carries its own state and agency.

```mermaid
classDiagram
    direction LR
    class AOClass
    class AOInstance {
        +invoke() structured response
    }
    class Session {
        +is_active bool
    }

    AOClass "1" --> "*" AOInstance : instantiated into
    AOInstance "1" --> "*" Session : owner
    AOInstance ..> Session : "Invocation creates"
```

### Sessions

Just as classes are instantiated into objects, objects are invoked into sessions. Every agentic object can maintain **multiple sessions concurrently**, and sessions are identified by their thread ID (or thread name).

A session belongs exclusively to the agentic object that created it. The same thread ID on two different agentic objects yields two different sessions.

## Invoking Agentic Objects

When you invoke an agentic object, you specify whether to:

- **Create a new session** — a fresh conversation context
- **Reuse an existing session** — load a session by its thread ID

The invocation determines the session lifecycle; the agentic object manages the rest.

## Call Stacks Across Objects

An agentic object acting within a session can itself invoke other agentic objects — on its own class or on different ones. This creates a **traceback** that spans across multiple objects and sessions, analogous to a call stack in traditional code execution.

```mermaid
sequenceDiagram
    participant Caller
    participant A as AO-A (Session-A)
    participant B as AO-B (Session-B)

    Caller->>A: invoke()
    A->>A: create or load session
    A->>B: invoke()
    B-->>A: result
    A-->>Caller: result
```

Each invocation in the traceback creates or reuses a session on the target agentic object. When sessions are materialized on disk (along with their contexts), the full invocation chain becomes traceable — the chat history can be walked back across object boundaries and session boundaries.

### The No-Circle Restriction

Sessions must never be invoked concurrently. If an agentic object is already working inside a session, that session is **active**. Invoking an active session is not allowed.

This means the invocation traceback must be **acyclic** with respect to sessions — no session UID may appear more than once along any invocation chain. The implementation enforces this by detecting an attempt to invoke an active session and returning an error to the caller.
