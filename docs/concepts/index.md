# Concepts

Deep dives into the core ideas behind sOAP: how agentic objects are composed, invoked, tested, and persisted.

## Topics

| Topic | Description |
|---|---|
| [Agentic Object Roles](./agentic-object-roles.md) | Role naming, canonical role building, user overrides, model selection |
| [Composition](./composition.md) | Multi-inheritance, MRO-based system prompts, config merging across agentic parents |
| [Adaptive Objects](./adaptive-objects.md) | Objects that manipulate and extend their own code at runtime across class, instance, and session levels |
| [Invocation](./invocation.md) | `invoke_agent()`, `invoke()`, output schemas, thread IDs, sub-agents |
| [Steering the Agent](./steering.md) | Tool return values and queued messages — two ways to guide the agent mid-invocation |
| [Invocation Hooks](./invocation-hooks.md) | Observing, controlling, and measuring agent invocations |
| [State and Persistence](./state-and-persistence.md) | Internal state, session serialization, context forking, token counting, compaction |
| [Media Handling](./media.md) | Attaching images, video, and PDFs to prompts; agent-initiated media access |
| [Testing](./testing.md) | Classical unit tests for deterministic code, Monte Carlo testing for agent behavior |
