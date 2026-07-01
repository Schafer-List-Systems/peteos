# ContentMedia

```python
from peteos.conversation.media import (
    create_media_content_part,
    create_media_content_part_async,
    create_image_content_part,
    create_image_content_part_async,
    create_content_part,
    create_content_part_async,
)
```

Utility functions that convert file paths, raw bytes, or HTTP(S) URLs into `ContentPart` objects for inclusion in chatbot messages.

## Synchronous Functions

```python
create_media_content_part(src: str | bytes, mime_type: str | None = None) -> ContentPart
```

Create a ContentPart from a local file path or raw bytes.

| Parameter | Type | Description |
|---|---|---|
| `src` | `str \| bytes` | Local file path, or raw media bytes. |
| `mime_type` | `str \| None` | Required when `src` is bytes. |

**Returns:** `ContentPart` with the appropriate part_type (image/video/pdf).

```python
create_image_content_part(src: str) -> ContentPart
```

Create a ContentPart for an image from a local file path.

```python
create_content_part(src: str) -> ContentPart
```

Create a ContentPart for any file type from a local file path (image, video, or PDF).

## Asynchronous Functions

```python
create_media_content_part_async(src: str | bytes, mime_type: str | None = None, timeout: float = 30.0) -> ContentPart
```

Like `create_media_content_part` but supports URL fetching.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `src` | `str \| bytes` | — | File path, raw bytes, or HTTP(S) URL. |
| `mime_type` | `str \| None` | `None` | Required when `src` is bytes. |
| `timeout` | `float` | `30.0` | Request timeout in seconds (URL fetching only). |

```python
create_image_content_part_async(src: str, timeout: float = 30.0) -> ContentPart
```

Like `create_image_content_part` but supports URL fetching.

```python
create_content_part_async(src: str, timeout: float = 30.0) -> ContentPart
```

Like `create_content_part` but supports URL fetching.

## MIME Type Constants

The following extension-to-MIME mappings are used for content detection:

| Group | Extensions |
|---|---|
| **IMAGE** | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.svg`, `.ico`, `.avif` |
| **VIDEO** | `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv`, `.flv`, `.m4v`, `.wmv` |
| **DOCUMENT** | `.pdf`, `.doc`, `.docx`, `.odt`, `.rtf`, `.xls`, `.xlsx`, `.csv`, `.ppt`, `.pptx`, `.md`, `.txt`, `.json`, `.xml`, `.html` |

## Usage

Attach media to an agent invocation via the `image` parameter on `invoke_agent()`, or queue media as messages using `ContentPart` + `Message.create()`:

```python
from peteos.conversation.media import create_content_part_async
from peteos.conversation import Message, ContentPart

media_part = await create_content_part_async("photo.jpg")
msg = Message.create(
    role="user",
    content_parts=[
        ContentPart.create_text("What do you see?"),
        media_part,
    ],
)
```
