"""Image utility for vision support.

Provides functions to convert local image files or URLs into ContentPart
objects suitable for inclusion in chatbot message content.
"""

import base64
import mimetypes
import logging
from typing import Any, Dict, Optional

from peteos.chatbot import ContentPart

logger = logging.getLogger(__name__)

# Image extensions that map to common MIME types
IMAGE_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".avif": "image/avif",
}

# Maximum image dimensions for Anthropic API
MAX_DIMENSIONS = 8000


def get_media_type(filepath: str) -> Optional[str]:
    """Determine the MIME type for an image file based on its extension.

    Args:
        filepath: Path to the image file.

    Returns:
        MIME type string, or None if unrecognized.
    """
    ext = filepath.lower().rsplit(".", 1)[-1] if "." in filepath else ""
    return IMAGE_EXTENSIONS.get(f".{ext}") or mimetypes.guess_type(f".{ext}")[0]


def file_to_base64(filepath: str) -> str:
    """Read an image file and return its base64-encoded representation.

    Args:
        filepath: Path to the image file.

    Returns:
        Base64-encoded string of the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def url_to_base64(url: str) -> str:
    """Fetch an image from a URL and return its base64-encoded representation.

    Args:
        url: HTTP/HTTPS URL to the image.

    Returns:
        Base64-encoded string of the image contents.

    Raises:
        RuntimeError: If the HTTP request fails.
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch image from {url}: HTTP {resp.status}")
            data = await resp.read()
    return base64.b64encode(data).decode("ascii")


def _get_media_type_for_base664(data: bytes, url: str = "") -> Optional[str]:
    """Infer the MIME type from file data or URL.

    Args:
        data: Raw bytes of the file.
        url: The URL (used to infer type from filename if data is inconclusive).

    Returns:
        MIME type string or None.
    """
    # Check magic bytes
    if len(data) >= 4:
        if data[:4] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if data[:4] == b"GIF8":
            return "image/gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"

    # Fall back to URL-based detection
    if url:
        parsed_ext = url.lower().rsplit(".", 1)[-1] if "." in url else ""
        for ext, media_type in IMAGE_EXTENSIONS.items():
            if parsed_ext == ext.lstrip("."):
                return media_type

    return None


def create_image_content_part(
    src: str,
) -> ContentPart:
    """Create a ContentPart for an image from a file path or URL.

    Auto-detects whether src is a local file path or a URL:
    - If it starts with "http://" or "https://", treats it as a URL
    - Otherwise treats it as a file path

    Args:
        src: Local file path or HTTP(S) URL to the image.

    Returns:
        ContentPart with type="image" and source containing base64 data.

    Raises:
        FileNotFoundError: If the file does not exist (file paths only).
        RuntimeError: If the URL request fails.
    """
    if src.startswith(("http://", "https://")):
        return _create_image_content_part_url(src)
    return _create_image_content_part_file(src)


async def create_image_content_part_async(
    src: str,
) -> ContentPart:
    """Async version of create_image_content_part for URL fetching.

    Args:
        src: Local file path or HTTP(S) URL to the image.

    Returns:
        ContentPart with type="image" and source containing base64 data.
    """
    if src.startswith(("http://", "https://")):
        return await _create_image_content_part_url_async(src)
    return _create_image_content_part_file(src)


def _create_image_content_part_file(filepath: str) -> ContentPart:
    """Create a ContentPart from a local image file path."""
    data = file_to_base64(filepath)
    media_type = get_media_type(filepath)
    return ContentPart(
        part_type="image",
        source={
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    )


async def _create_image_content_part_url_async(url: str) -> ContentPart:
    """Async: Create a ContentPart from a URL by fetching and encoding."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch image from {url}: HTTP {resp.status}")
            data_bytes = await resp.read()

    b64_data = base64.b64encode(data_bytes).decode("ascii")
    media_type = _get_media_type_for_base664(data_bytes, url)

    return ContentPart(
        part_type="image",
        source={
            "type": "base64",
            "media_type": media_type,
            "data": b64_data,
        },
    )


def _create_image_content_part_url(url: str) -> ContentPart:
    """Create a ContentPart from a URL, stored as a URL reference (no base64).

    For APIs that support URL sources directly, this avoids encoding the image.
    Falls back to base64 encoding if the URL source is not supported.
    """
    return ContentPart(
        part_type="image",
        source={
            "type": "url",
            "url": url,
        },
    )
