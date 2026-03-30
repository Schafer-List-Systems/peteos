# Testing Guide

## Overview

The peteos project includes unit tests and end-to-end (E2E) tests that verify the framework's core components. Tests use `pytest` with `pytest-asyncio` for async test support.

## Running Tests

### Run All Tests

```bash
pytest -v
```

### Run Specific Test Files

```bash
# Unit tests
pytest tests/unit/ -v

# ChatBot response tests
pytest tests/test_chatbot_response.py -v

# E2E integration tests
pytest tests/e2e/ -v

# HTTPClient tests
pytest tests/test_httpclient.py -v
```

### Run Tests with Coverage

```bash
pytest --cov=peteos --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_chatbot_response.py::test_anthropic_compatible_thinking -v
```

## Test Structure

```
tests/
├── __init__.py
├── http/
│   ├── __init__.py
│   └── mock_server.py      # Mock HTTP servers for testing
├── unit/
│   ├── __init__.py
│   ├── mock_httpclient.py          # Mock HTTP client
│   ├── test_chatbot_manager.py     # ChatBotManager tests
│   ├── test_chatbot_response.py    # ChatBotResponse tests
│   ├── test_chathistory.py         # ChatHistory tests
│   ├── test_httpclient.py          # HTTPClient tests
│   ├── test_replexecutionenvironment.py  # REPLExecutionEnvironment tests
│   ├── test_role.py                # Role tests
│   ├── test_rolemanager.py         # RoleManager tests
│   ├── test_session.py             # Session tests
│   └── test_toolmanager.py         # ToolManager tests
└── e2e/
    ├── __init__.py
    └── test_chatbot_integration.py       # ChatBot integration tests
```

## Mock Infrastructure

### Mock HTTP Servers

`tests/http/mock_server.py` provides mock servers for testing ChatBot implementations:

```python
from tests.http.mock_server import create_openai_mock_server, create_anthropic_mock_server

# OpenAI-compatible mock server
server = create_openai_mock_server(
    reasoning="I'm thinking...",
    response="Hello, world!",
    port=8765
)
await server.serve()

# Anthropic-compatible mock server
server = create_anthropic_mock_server(
    thinking="I'm thinking...",
    response="Hello from Anthropic!",
    port=8766
)
await server.serve()
```

### Mock HTTP Client

`tests/mock_httpclient.py` provides a mock HTTP client for testing without real API calls:

```python
from tests.mock_httpclient import MockHTTPClient

mock_client = MockHTTPClient()
mock_client.add_response({"choices": [{"delta": {"content": "test"}}]})
```

## Test Coverage

### ChatBotResponse Tests (`tests/test_chatbot_response.py`)

Tests for response parsing and translation:
- `test_openai_delta_streaming`: OpenAI streaming response parsing
- `test_openai_message_streaming`: OpenAI message streaming with `message.content`
- `test_anthropic_streaming`: Anthropic streaming response parsing
- `test_anthropic_compatible_thinking`: Anthropic-compatible reasoning detection
- `test_anthropic_non_streaming`: Anthropic non-streaming response handling
- `test_wildcard_path_translation`: Wildcard path translation (`choices[*]`)
- `test_type_discriminator`: Type discriminator path translation
- `test_multiple_fields_same_event`: Multi-field event handling

### E2E Integration Tests (`tests/e2e/test_chatbot_integration.py`)

End-to-end tests with mock servers:
- `test_openai_chatbot_streaming_with_mock`: Full OpenAI ChatBot streaming integration
- `test_openai_chatbot_non_streaming_with_mock`: Full OpenAI ChatBot non-streaming integration
- `test_anthropic_chatbot_streaming_with_mock`: Full Anthropic ChatBot streaming integration
- `test_anthropic_chatbot_non_streaming_with_mock`: Full Anthropic ChatBot non-streaming integration
- `test_anthropic_chatbot_with_reasoning`: Anthropic reasoning content handling

### REPLExecutionEnvironment Tests (`tests/unit/test_replexecutionenvironment.py`)

Tests for the REPL agentic loop:
- `test_basic_conversation`: Basic conversation flow
- `test_tool_call_detection`: Tool call detection from response
- `test_tool_call_execution`: Tool call execution and loop continuation
- `test_tool_call_not_found`: Tool not found error handling
- `test_tool_call_exception`: Tool execution exception handling
- `test_loop_termination`: Loop termination on final answer
- `test_interrupt_handling`: Interrupt handling during streaming
- `test_thinking_content_accumulation`: Thinking content handling

### Other Unit Tests

- `tests/unit/test_chathistory.py`: ChatHistory message management
- `tests/unit/test_httpclient.py`: HTTPClient streaming and non-streaming
- `tests/unit/test_role.py`: Role loading from dict and path
- `tests/unit/test_rolemanager.py`: RoleManager registration, directory loading, and lookup
- `tests/unit/test_session.py`: Session initialization, load_from_json, load_from_file
- `tests/unit/test_chatbot_manager.py`: ChatBotManager backend management, model discovery
- `tests/unit/test_toolmanager.py`: Tool registration and execution

## Writing New Tests

### Unit Test Example

```python
import pytest
from peteos.chatbotresponse import GenericChatBotResponse

@pytest.mark.asyncio
async def test_basic_translation():
    """Test basic path translation."""
    async def mock_stream():
        yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        yield '[DONE]'

    translations = {"choices[*].delta.content": "text"}
    response = GenericChatBotResponse(mock_stream(), translations)

    async for key, chunk in response:
        assert key == "text"
        assert chunk == "Hello"

    assert response.data["text"] == "Hello"
```

### E2E Test Example

```python
import pytest
from peteos.chatbot import OpenAIChatBot
from peteos.httpclient import HTTPClient
from peteos.chathistory import ChatHistory
from peteos.message import Message
from tests.http.mock_server import create_openai_mock_server

@pytest.mark.asyncio
async def test_chatbot_with_mock_server():
    """Test ChatBot with mock server."""
    # Start mock server
    server = create_openai_mock_server(
        reasoning="Thinking...",
        response="Hello!",
        port=8765
    )
    server_task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(0.1)

    try:
        # Create ChatBot
        http_client = HTTPClient(timeout=60.0)
        chatbot = OpenAIChatBot(
            http_client=http_client,
            model="test-model",
            base_url="http://localhost:PORT"
        )

        # Send message
        history = ChatHistory()
        history.append_message(Message(content={"role": "user", "content": "Hi"}))
        response = await chatbot.send_message(history, streaming=True)

        # Verify response
        async for key, chunk in response:
            if key == "text":
                assert "Hello" in chunk

        assert "Thinking..." in response.data.get("reasoning", "")
    finally:
        server_task.cancel()
```

## Testing Best Practices

1. **Use Mock Servers**: Prefer `tests/http/mock_server.py` over real API calls
2. **Async Tests**: Use `@pytest.mark.asyncio` for all async test functions
3. **Cleanup**: Always clean up mock servers and resources in `finally` blocks
4. **Isolation**: Each test should be independent and not share state
5. **Coverage**: Test both streaming and non-streaming modes

## Test Dependencies

```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "aiohttp"]
```

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

## CI/CD Integration

Tests should run automatically on:
- Pull requests to `main`
- Daily builds
- Manual triggers via GitHub Actions

Example workflow (`.github/workflows/test.yml`):
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pytest -v --cov=peteos
```
