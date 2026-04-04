"""Logging configuration for Peteos."""

import logging
import sys
from typing import Optional


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


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)


# Create a debug logger for specific modules
class DebugLogger:
    """Logger that only outputs when debug mode is enabled."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._enabled = False

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug output."""
        self._enabled = enabled
        if enabled:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)

    def _log(self, level: int, message: str, *args, **kwargs) -> None:
        """Log message only if debug mode is enabled."""
        if self._enabled:
            self._logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, *args, **kwargs)


def create_debug_logger(name: str) -> DebugLogger:
    """Create a debug logger that respects debug mode flag.

    Args:
        name: Logger name.

    Returns:
        A DebugLogger instance.
    """
    return DebugLogger(name)
