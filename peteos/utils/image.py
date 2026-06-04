"""Image utility for vision support.

Provides functions to convert local image files or URLs into ContentPart
objects suitable for inclusion in chatbot message content.
"""

import asyncio
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

VIDEO_EXTENSIONS = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".flv": "video/x-flv",
    ".m4v": "video/x-m4v",
    ".wmv": "video/x-wmv",
}

DOCUMENT_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".rtf": "application/rtf",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".csv": "text/csv",
    ".tsv": "text/tsv",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".xml": "application/xml",
    ".html": "text/html",
}

# Maximum image dimensions for Anthropic API
MAX_DIMENSIONS = 8000


def _ext_map(filepath: str) -> str:
    """Extract lowercase file extension from a filepath.

    Args:
        filepath: Path to the file.

    Returns:
        Extension string including the dot, or empty string.
    """
    ext = filepath.lower().rsplit(".", 1)[-1] if "." in filepath else ""
    return f".{ext}"


def get_media_type(filepath: str) -> Optional[str]:
    """Determine the MIME type for any file based on its extension.

    Checks image, video, and document extensions in order.

    Args:
        filepath: Path to the file.

    Returns:
        MIME type string, or None if unrecognized.
    """
    ext = _ext_map(filepath)
    for ext_map in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, DOCUMENT_EXTENSIONS):
        result = ext_map.get(ext)
        if result:
            return result
    return mimetypes.guess_type(f".{ext}")[0]


def get_content_type(filepath: str) -> Optional[str]:
    """Determine the uniform ContentPart type for any file.

    Args:
        filepath: Path to the file.

    Returns:
        One of "image", "video", "pdf", or None.
    """
    ext = _ext_map(filepath)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in DOCUMENT_EXTENSIONS:
        return "pdf"
    return None


def file_to_base64(filepath: str) -> str:
    """Read a file and return its base64-encoded representation.

    Args:
        filepath: Path to the file.

    Returns:
        Base64-encoded string of the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def url_to_base64(url: str, timeout: float) -> str:
    """Fetch an image from a URL and return its base64-encoded representation.

    Args:
        url: HTTP/HTTPS URL to the image.
        timeout: Request timeout in seconds. Must be provided explicitly.

    Returns:
        Base64-encoded string of the image contents.

    Raises:
        RuntimeError: If the HTTP request fails.
    """
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to fetch image from {url}: HTTP {resp.status}")
                data = await resp.read()
        return base64.b64encode(data).decode("ascii")
    except asyncio.TimeoutError:
        raise RuntimeError(f"Timeout fetching image from {url} after {timeout}s")


def _get_media_type_for_base64(data: bytes, url: str = "") -> Optional[str]:
    """Infer the MIME type from file data or URL.

    Checks magic bytes first, then falls back to URL extension detection
    across all supported file types.

    Args:
        data: Raw bytes of the file.
        url: The URL (used to infer type from filename if data is inconclusive).

    Returns:
        MIME type string or None.
    """
    if len(data) >= 4:
        if data[:4] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if data[:4] == b"GIF8":
            return "image/gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        if data[:4] == b"RIFF" and data[8:12] == b"AVIF":
            return "image/avif"

    # Fall back to URL-based detection
    if url:
        parsed_ext = url.lower().rsplit(".", 1)[-1] if "." in url else ""
        for ext_map in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, DOCUMENT_EXTENSIONS):
            for ext, media_type in ext_map.items():
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
    timeout: float = None,
) -> ContentPart:
    """Async version of create_image_content_part for URL fetching.

    Args:
        src: Local file path or HTTP(S) URL to the image.
        timeout: Request timeout in seconds.

    Returns:
        ContentPart with type="image" and source containing base64 data.
    """
    if src.startswith(("http://", "https://")):
        return await _create_image_content_part_url_async(src, timeout)
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


async def _create_image_content_part_url_async(url: str, timeout: float) -> ContentPart:
    """Async: Create a ContentPart from a URL by fetching and encoding."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch image from {url}: HTTP {resp.status}")
            data_bytes = await resp.read()

    b64_data = base64.b64encode(data_bytes).decode("ascii")
    media_type = _get_media_type_for_base64(data_bytes, url)

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


def create_content_part(src: str) -> ContentPart:
    """Create a ContentPart for any file type from a file path or URL.

    Auto-detects the file type by extension and creates the appropriate
    uniform ContentPart:

    - **Images** (.png, .jpg, .gif, etc.): ``ContentPart(type="image")``
    - **Videos** (.mp4, .webm, .mov, etc.): ``ContentPart(type="video")``
    - **Documents** (.pdf, .docx, .xlsx, etc.): ``ContentPart(type="pdf")``

    For images, base64 encoding is used. For videos and documents, the file
    is also base64-encoded but the uniform type signals to the translation
    layer which API format to use.

    Auto-detects whether src is a local file path or a URL:
    - If it starts with "http://" or "https://", treats it as a URL
    - Otherwise treats it as a file path

    Args:
        src: Local file path or HTTP(S) URL to the file.

    Returns:
        ContentPart with the appropriate uniform type and source.

    Raises:
        FileNotFoundError: If the file does not exist (file paths only).
        RuntimeError: If the URL request fails.
    """
    if src.startswith(("http://", "https://")):
        return _create_content_part_url(src)
    return _create_content_part_file(src)


async def create_content_part_async(src: str, timeout: float) -> ContentPart:
    """Async version of create_content_part for URL fetching.

    Args:
        src: Local file path or HTTP(S) URL to the file.
        timeout: Request timeout in seconds.

    Returns:
        ContentPart with the appropriate uniform type and source.
    """
    if src.startswith(("http://", "https://")):
        return await _create_content_part_url_async(src, timeout)
    return _create_content_part_file(src)


def _create_content_part_file(filepath: str) -> ContentPart:
    """Create a ContentPart from a local file path."""
    content_type = get_content_type(filepath)
    if content_type is None:
        # Unknown type, fall back to image
        return _create_image_content_part_file(filepath)

    media_type = get_media_type(filepath)
    data = file_to_base64(filepath)
    return ContentPart(
        part_type=content_type,
        source={
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    )


async def _create_content_part_url_async(url: str, timeout: float) -> ContentPart:
    """Async: Create a ContentPart from a URL by fetching and encoding."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch file from {url}: HTTP {resp.status}")
            data_bytes = await resp.read()

    b64_data = base64.b64encode(data_bytes).decode("ascii")
    content_type = get_content_type(url)
    if content_type is None:
        content_type = "image"
    media_type = _get_media_type_for_base64(data_bytes, url)

    return ContentPart(
        part_type=content_type,
        source={
            "type": "base64",
            "media_type": media_type,
            "data": b64_data,
        },
    )


def _create_content_part_url(url: str) -> ContentPart:
    """Create a ContentPart from a URL, stored as a URL reference (no base64).

    For APIs that support URL sources directly, this avoids encoding the file.
    """
    content_type = get_content_type(url)
    if content_type is None:
        content_type = "image"
    return ContentPart(
        part_type=content_type,
        source={
            "type": "url",
            "url": url,
        },
    )
