# Vision (Image Support)

## Source

- [Anthropic API: Images and Vision](https://platform.claude.com/docs/en/build-with-claude/vision)

## Overview

Claude's vision capabilities allow the agent to understand and analyze images through the Anthropic API. Images are stored as `ContentPart` objects with `type="image"` and a `source` dictionary in the message content array.

## Image Formats

The Anthropic Messages API supports three ways to include images in messages:

### 1. Base64 Encoded Images

Embed the image directly in the API request by encoding it as base64:

```python
ContentPart(
    type="image",
    source={
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgoAAAANSUhEUg..."  # base64 string
    }
)
```

Corresponding API JSON:
```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "iVBORw0KGgo..."
  }
}
```

**How peteos handles it**: `peteos/utils/image.py::create_image_content_part()` reads the file or URL, encodes to base64, and creates the ContentPart with the correct `media_type`.

### 2. URL-Based Images

Reference an image by its URL (the API fetches it directly):

```python
ContentPart(
    type="image",
    source={
        "type": "url",
        "url": "https://example.com/image.png"
    }
)
```

Corresponding API JSON:
```json
{
  "type": "image",
  "source": {
    "type": "url",
    "url": "https://example.com/image.png"
  }
}
```

**How peteos handles it**: URL-based images are stored directly as the `url` in `source`, avoiding base64 encoding. When sent to the Anthropic API, the source is passed through unchanged.

### 3. Files API (file_id)

Upload images to the Anthropic Files API and reference by `file_id`:

```python
ContentPart(
    type="image",
    source={
        "type": "ebs",
        "file_id": "file_abc123"
    }
)
```

Corresponding API JSON:
```json
{
  "type": "image",
  "source": {
    "type": "ebs",
    "file_id": "file_abc123"
  }
}
```

Use this format for many or large images to keep request payloads under the 32MB limit.

## API Limits

| Limit | 200k context models | Other models |
|-------|--------------------|--------------|
| Images per message | 20 | 20 |
| Images per request | 100 | 600 |
| Max dimensions per image | 8000x8000 px | 8000x8000 px |
| Max dimensions (>20 images) | 2000x2000 px | 2000x2000 px |
| Max request size | 32 MB | 32 MB (lower on Bedrock/Vertex AI) |

## Token Counting for Images

**How peteos handles it**: `Message._compute_token_count()` serializes image ContentParts to JSON including their base64 data, so the token count accurately reflects the data volume the LLM will process. This ensures compaction triggers at the correct time, even with large images.

## Architecture

### Message Storage

Images are stored as `ContentPart` objects in a `Message`'s content array:

```python
from peteos.chatbot import Message, ContentPart

msg = Message(
    role="user",
    content=[
        ContentPart(type="text", text="What's in this image?"),
        ContentPart(type="image", source={"type": "base64", "media_type": "image/png", "data": "..."})
    ]
)
```

### API Serialization

- **Anthropic API**: ContentParts with `type="image"` pass through `_build_body()` unchanged, maintaining the correct format
- **OpenAI API**: ContentParts are converted to `{"type": "image_url", "image_url": {"url": "..."}}` format in the content array

### User Input (Shell Channel)

The shell channel provides a `/image` command:

```
> /image /path/to/photo.png Describe this image
> [Image] Describe this image
```

The command reads the file, encodes it to base64, and queues the message to the session.

### Display

`Message.printable()` renders `[Image]` placeholders for non-text parts:

```
What's in this image? [Image]
```

## Code Examples

### Reading a local image file

```python
from peteos.utils.image import create_image_content_part

part = create_image_content_part("/path/to/photo.png")
# Returns ContentPart(type="image", source={"type": "base64", "media_type": "image/png", "data": "..."})
```

### Creating a message with text + image

```python
from peteos.chatbot import Message, ContentPart

msg = Message(
    role="user",
    content=[
        ContentPart(type="text", text="What country is this?"),
        create_image_content_part("/path/to/photo.jpg")
    ]
)
```

### Sending to Anthropic API

```python
from peteos.chatbot import AnthropicChatBot, ChatHistory

history = ChatHistory()
history.append_message(msg)

response = await chatbot.send_message(history)
# Anthropic API receives the image in the content array
```
