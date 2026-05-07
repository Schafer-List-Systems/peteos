# Logging in Peteos

## Overview

Peteos uses Python's standard `logging` module for all logging output. This provides:

- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Support for logging to stdout, stderr, or files
- Debug mode for detailed diagnostic output
- Structured logging with timestamps and module names

## Usage

### Basic Setup

```python
from peteos.logger import setup_logging, get_logger

# Configure logging
setup_logging(level="INFO", debug=False, log_to_file=None)

# Get a logger
logger = get_logger(__name__)

# Use logging methods
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Log Levels

| Level | Description | Default | Debug Mode |
|-------|-------------|---------|------------|
| DEBUG | Detailed diagnostic information | No | Yes |
| INFO | General information about operation | Yes | Yes |
| WARNING | Potential issues | Yes | Yes |
| ERROR | Error conditions | Yes | Yes |
| CRITICAL | Severe errors | Yes | Yes |

### Configuration Options

```python
setup_logging(
    level="INFO",           # Base log level
    debug=False,            # Enable additional debug output
    log_to_file=None       # Optional file path for logs
)
```

### Examples

**Console output (default):**
```bash
python examples/logging_demo.py
```

**Debug output:**
```bash
python examples/logging_demo.py --debug
```

**File output:**
```bash
python examples/logging_demo.py --log-file /tmp/peteos.log
```

## Best Practices

1. **Use appropriate log levels**: 
   - DEBUG for detailed tracing (development only)
   - INFO for normal operations
   - WARNING for recoverable issues
   - ERROR for errors that prevent operation
   - CRITICAL for severe errors

2. **Don't use print()**: Use the logger instead for consistent output

3. **Include context**: Add relevant details to messages for easier debugging

4. **Keep modules quiet**: Submodules like `aiohttp` and `httpx` are configured to log at WARNING level by default to reduce noise
