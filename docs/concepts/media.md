# Media Handling

Agentic objects can receive images as input (the agent sees them), produce images (the developer injects them into the agent's context), and reason about them (the agent independently loads and analyzes them). All three flows are supported for image content and are enabled by different capabilities and configuration.

## Prerequisite: A Vision-Capable Backend

Media handling requires an LLM backend that supports vision. The chosen model must be able to process base64-encoded image content blocks. Examples include vision-capable proprietary models and open source models such as Qwen 3.6 with vision support.

## Media Input: User to Agent

The simplest way to give the agent access to an image is to pass it alongside the prompt during invocation:

```python
result = await obj.invoke_agent(
    prompt="What do you see in this image?",
    image="/path/to/screenshot.png",
)
```

The `image` parameter accepts a local file path or an HTTP(S) URL. The image is sent to the LLM as part of the user message.

Supported MIME types:

- **Images**: `image/png`, `image/jpeg`, `image/webp`

## Media Output: Developer to Agent

An agentic object can push images *into* the agent's context during invocation. This is how a tool implemented by the developer can send an image to the agent for analysis:

```python
class ImageAnalyzer(AgenticObject):
    """You analyze images sent by the user."""

    @tool(description="Send the processed image to the agent for analysis.")
    async def send_image_to_agent(self, runner) -> str:
        # ... process image, get PNG bytes ...
        await self._send_media(
            data=png_bytes,
            mime_type="image/png",
            runner=runner,
        )
        return "OK: Image sent to agent."
```

The `runner` argument is automatically injected by the framework and must be declared in the tool signature. The developer provides raw bytes and a MIME type. The framework injects the image into the agent's reasoning loop. The agent will then see the image in its next iteration.

## Self-Initiated Media Access: Agent to Itself

When `allow_media_access=True` is set on the agentic class (via the `@agentic_object` decorator), the agent can independently decide to load and reason about images during its own reasoning process:

```python
@agentic_object(allow_media_access=True)
class ImageAnalyst(AgenticObject):
    """You can read and reason about images. Use read_media to inspect files."""
```

When this flag is enabled, the agent can independently load and reason about images. It decides when to call `read_media` as part of its own reasoning:

```python
agent = ImageAnalyst()
result = await agent.invoke_agent("What's in the flowers.png?")
# -> "I can see roses in a vase."
```

The agent sees the prompt, decides to load `flowers.png`, reads it, and then reasons about its contents.

The `allow_media_access` flag is unioned across the entire MRO — if *any* parent class in the inheritance chain sets it to `True`, the tool is registered.

## The Media Flow

These three pathways form a complete image loop:

```
User provides image ──→ invoke_agent(image=...) ──→ Agent sees image
                                                   │
                                           Agent reasons
                                                   │
                                           Agent calls read_media(src=...) ──→ More image injected
                                                   │
                                           Agent calls tool ──→ _send_media(data=...) ──→ New image injected
```

The agent can cycle through self-initiated image loading and developer-provided images as many times as needed before producing its final response. Each injected image becomes part of the conversation history and is available for future invocations if the session is persistent.
