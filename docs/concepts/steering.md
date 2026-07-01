# Steering the Agent

When `invoke_agent()` runs, the agent reasons in a loop until it returns its result.
You can steer this loop from inside tool methods — the agent receives your input either as a **tool result** or as a **new message**, and from its perspective these are two very different signals.

## Why

A tool result is the agent's answer to a question it just asked itself.
It continues reasoning and considers the result as the completion of that call.
A queued message provides further information that is not tied to any tool call.
It shapes the agent's next reasoning step without interrupting anything —
the agent simply continues its loop and factors the new information into its next LLM response.
Which you use depends on whether the information completes a tool call or adds independent context.

## Tool Return Values

Every `@tool` decorated method returns a value that the agent sees as a **tool result**.
The return value is injected directly as the answer to the tool call the agent just made.
This is the primary steering channel and should be used whenever the information completes the tool's purpose.

```python
@tool
def check_inventory(self, item: str) -> str:
    """Check stock levels for an item."""
    qty = self._stock.get(item, 0)
    if qty == 0:
        return f"No stock available for '{item}'."
    return f"Stock for '{item}': {qty} units."
```

The agent sees this as the response to its `check_inventory` call and adjusts its reasoning accordingly.

## Queuing New Messages

You can also push new messages into the runner's event queue from inside a tool method.
This requires the `runner` argument — PeteOS auto-injects it when you add `runner: "Runner | None" = None` at the end of a tool's signature.

```python
@tool
async def analyze_image(self, image_path: str, runner: "Runner | None" = None) -> str:
    """Analyze an image and return the description. If the image is unclear, request additional photos from the user."""
    if runner is None:
        return "Runner not available."

    description = await self._process_image(image_path)
    if not description:
        from peteos.conversation import Message, ContentPart

        await runner.queue_message(
            Message.create(
                role="user",
                content_parts=[ContentPart.create_text(
                    "The image was unclear. Please provide a better photo."
                )],
            )
        )
        return "Requested a better photo."

    return f"Image analysis: {description}"
```

## When to Use Which

| Scenario | Mechanism | Why |
|---|---|---|
| Tool completes normally | Tool return value | Agent expects this as the tool's answer |
| Tool completes with an error or hint to retry | Tool return value (error message) | Agent already expects this tool's result |
| Provide context not tied to a tool call | `runner.queue_message()` | Independent information, shapes next reasoning step |

## Key Points

- The `runner` argument is auto-injected by PeteOS when declared in a tool's signature.
- Always check `if runner is None` — the runner may be absent in non-invocation contexts.
- Tool return values are always available and require no extra code.
- Queued messages go into the runner's event queue and are drained before the next LLM call.
