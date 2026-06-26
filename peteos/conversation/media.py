"""Media utility for multi-modal support.

Provides functions to convert file paths, URLs, or raw bytes into ContentPart
objects suitable for inclusion in chatbot message content.
"""

import base64
import logging
import mimetypes
from typing import Optional, Union

from peteos.conversation.message import ContentPart

logger = logging.getLogger(__name__)

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
    ".mkv": "video/x-matrosvid",
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


def _ext_map(filepath: str) -> str:
    """Extract lowercase file extension from a filepath."""
    ext = filepath.lower().rsplit(".", 1)[-1] if "." in filepath else ""
    return f".{ext}"


def get_media_type(filepath: str) -> Optional[str]:
    """Determine the MIME type for any file based on its extension."""
    ext = _ext_map(filepath)
    for ext_map in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, DOCUMENT_EXTENSIONS):
        result = ext_map.get(ext)
        if result:
            return result
    return mimetypes.guess_type(f".{ext}")[0]


def get_content_type(filepath: str) -> Optional[str]:
    """Determine the uniform ContentPart type for any file.

    Returns one of 'image', 'video', 'pdf', or None.
    """
    ext = _ext_map(filepath)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in DOCUMENT_EXTENSIONS:
        return "pdf"
    return None


def _file_to_base64(filepath: str) -> str:
    """Read a file and return its base64-encoded representation."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def _url_to_bytes(url: str, timeout: float) -> bytes:
    """Fetch a URL and return its raw bytes."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch from {url}: HTTP {resp.status}")
            return await resp.read()


def _get_media_type_for_base64(data: bytes, url_or_path: str = "") -> Optional[str]:
    """Infer the MIME type from file data or a path/URL.

    Checks magic bytes first, then falls back to extension detection.
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

    if url_or_path:
        parsed_ext = _ext_map(url_or_path)
        for ext_map in (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, DOCUMENT_EXTENSIONS):
            for ext, media_type in ext_map.items():
                if parsed_ext == ext:
                    return media_type
    return None


def _infer_part_type(media_type: Optional[str]) -> str:
    """Map a MIME type to a ContentPart part_type."""
    if not media_type:
        return "image"
    for part_type, ext_map in [("image", IMAGE_EXTENSIONS), ("video", VIDEO_EXTENSIONS), ("pdf", DOCUMENT_EXTENSIONS)]:
        for ext, mt in ext_map.items():
            if mt == media_type:
                return part_type
    return "image"


def _create_media_for_bytes(b64: str, media_type: Optional[str]) -> ContentPart:
    """Create the right ContentPart factory for a base64-encoded media source."""
    part_type = _infer_part_type(media_type)
    source = {"type": "base64", "media_type": media_type, "data": b64}
    if part_type == "video":
        return ContentPart.create_video(source)
    elif part_type == "pdf":
        return ContentPart.create_pdf(source)
    return ContentPart.create_image(source)


def create_media_content_part(
    src: str | bytes,
    mime_type: Optional[str] = None,
) -> ContentPart:
    """Create a ContentPart from a file path, raw bytes, or (async-only) URL.

    For synchronous use: accepts file paths (str) or raw bytes (bytes).
    For URL support, use ``create_media_content_part_async``.

    Args:
        src: Local file path (str) or raw media bytes (bytes).
        mime_type: Optional MIME type. Required when ``src`` is bytes.

    Returns:
        ContentPart with the appropriate part_type and base64-encoded source.

    Raises:
        FileNotFoundError: If src is a file path that does not exist.
        ValueError: If src is bytes and mime_type is not provided.
    """
    if isinstance(src, bytes):
        if mime_type is None:
            raise ValueError("mime_type is required when src is bytes")
        b64 = base64.b64encode(src).decode("ascii")
        return ContentPart.create_image(
            {"type": "base64", "media_type": mime_type, "data": b64}
        )
    return _create_media_content_part_file(src)


async def create_media_content_part_async(
    src: str | bytes,
    mime_type: Optional[str] = None,
    timeout: float = 30.0,
) -> ContentPart:
    """Create a ContentPart from a file path, raw bytes, or URL.

    Args:
        src: Local file path (str), raw bytes (bytes), or HTTP(S) URL (str).
        mime_type: Optional MIME type. Required when ``src`` is bytes.
        timeout: Request timeout in seconds (used only for URL fetching).

    Returns:
        ContentPart with the appropriate part_type and base64-encoded source.
    """
    if isinstance(src, bytes):
        if mime_type is None:
            raise ValueError("mime_type is required when src is bytes")
        b64 = base64.b64encode(src).decode("ascii")
        return _create_media_for_bytes(b64, mime_type)
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        data = await _url_to_bytes(src, timeout)
        media_type = _get_media_type_for_base64(data, src)
        b64 = base64.b64encode(data).decode("ascii")
        return _create_media_for_bytes(b64, media_type)
    return _create_media_content_part_file(src)


def _create_media_content_part_file(filepath: str) -> ContentPart:
    """Create a ContentPart from a local file path."""
    data = _file_to_base64(filepath)
    media_type = get_media_type(filepath)
    return _create_media_for_bytes(data, media_type)


# ── Backward-compatible aliases ──────────────────────────────────────
# These delegate to the new unified functions for drop-in replacement.

def create_image_content_part(src: str) -> ContentPart:
    """Create a ContentPart for an image from a file path."""
    return _create_media_content_part_file(src)


async def create_image_content_part_async(src: str, timeout: float = 30.0) -> ContentPart:
    """Async version of create_image_content_part for URL fetching."""
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        data = await _url_to_bytes(src, timeout)
        media_type = _get_media_type_for_base64(data, src)
        b64 = base64.b64encode(data).decode("ascii")
        return ContentPart.create_image(
            {"type": "base64", "media_type": media_type, "data": b64}
        )
    return create_image_content_part(src)


async def create_content_part_async(src: str, timeout: float = 30.0) -> ContentPart:
    """Async version of create_content_part for URL fetching."""
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        data = await _url_to_bytes(src, timeout)
        media_type = _get_media_type_for_base64(data, src)
        b64 = base64.b64encode(data).decode("ascii")
        return _create_media_for_bytes(b64, media_type)
    return _create_media_content_part_file(src)


def create_content_part(src: str) -> ContentPart:
    """Create a ContentPart for any file type from a file path."""
    return _create_media_content_part_file(src)
