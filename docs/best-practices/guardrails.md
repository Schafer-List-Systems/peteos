# Guardrails

Agents are non-deterministic — they make mistakes occasionally.
Guardrails are internal checks that validate an agent's output and retry when needed.

In the sOAP paradigm, guardrails are easy to implement because every agent call returns a value you can inspect.
Instead of intercepting the agent externally (which requires hooks, custom channels, or approval workflows), simply wrap the invocation in a method that validates the result and retries.

## Why

External intervention is expensive: hooks add complexity, custom channels are fragile, and approval workflows block the entire system.
When validation lives inside the agentic object itself, it's co-located with the invocation, runs only when needed, and doesn't affect other code paths.

## How

1. Call [`invoke_agent()`](../reference/agentic-object-base.md#invoke_agent) with an `output_schema` to get structured output.
2. Check the result against your criteria (not just the agent's word — verify side effects, data state, or schema validity).
3. If it fails, retry with a clarified prompt. Limit retries to avoid infinite loops.

```python
async def process_email(self):
    prompt = "Process the email and call produce_output."
    # 1. Loop until the agent actually does the right thing
    for attempt in range(3):
        status = await self.invoke_agent(
            prompt=prompt,
            output_schema=TaskStatus,
            persistent_thread_id="my-thread"
        )

        # 2. Validate: did the agent actually update state?
        if self._text and self._subject:
            break

        # 3. Retry with a reminder
        prompt = "You forgot to set the text and subject."

    # ... continue with validated result ...
```

## Edge Case: Structured Output Validation

When the agent returns a data structure, validate its contents — not just its type.
Feed error messages back to the agent so it can correct the mistake.

```python
for attempt in range(3):
    summary = await self.invoke_agent(
        prompt="Summarize the key topics and sentiment.",
        output_schema=TopicSummary,
    )
    if summary.topics and summary.sentiment in ("positive", "neutral", "negative"):
        break
    prompt = f"Your summary was incomplete: {summary}. Include all required fields."
```

## Key Principles

- **Validate side effects, not just return values.** Did the state actually change?
- **Limit retries.** 3 is a good default — it catches most mistakes without blocking forever.
- **Feed errors back.** Tell the agent what went wrong so it can self-correct on the next attempt.
- **Keep it internal.** Guardrails belong inside agentic object methods, not in external hooks.
