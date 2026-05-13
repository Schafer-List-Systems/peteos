"""Logging configuration for Peteos."""

import logging
import sys
from typing import Optional

_DEFAULT_TRUNC_LEN = 256


def truncate(value: str, max_len: int = _DEFAULT_TRUNC_LEN) -> str:
    """Truncate a string to *max_len* characters, appending ``[...]``.

    Returns the original string unchanged if it is already shorter than
    *max_len*.  The ``[...]`` sentinel signals to the reader that the
    value was cut off.

    Args:
        value: The string to truncate.
        max_len: Maximum length before truncation kicks in.

    Returns:
        The (possibly truncated) string.
    """
    if value is None:
        return value
    if len(value) <= max_len:
        return value
    return value[:max_len] + "[...]"


def setup_logging(
    level: str = "INFO",
    debug: bool = False,
    log_to_file: Optional[str] = None
) -> None:
    """Configure logging for Peteos.

    Args:
        level: Base log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        debug: Enable debug mode - adds more detailed logging.
        log_to_file: Optional path to log file. If None, logs to stdout.
    """
    # Determine effective level
    effective_level = logging.DEBUG if debug else level.upper()

    # Create formatter
    if debug:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%(levelname)s: %(message)s"
        date_fmt = None

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # Create handler
    if log_to_file:
        handler = logging.FileHandler(log_to_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Add our handler
    root_logger.addHandler(handler)

    # Configure aiohttp to be quieter by default
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
