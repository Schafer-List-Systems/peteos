# WebNavigator

A web navigator agentic object with tools to fetch, render, and screenshot web pages.

## Description

`WebNavigator` gives the agent the ability to retrieve and analyze web content at multiple tiers:

- **Fast HTTP fetch** — retrieve and parse static HTML pages
- **Headless Chrome rendering** — execute JavaScript for SPAs and dynamic content
- **Screenshot capture** — save a visual snapshot of a rendered page

## Tools

### `fetch_raw(url: str, timeout: int = 15) -> dict`

Fetch the raw HTTP response including headers and body.

Returns a dict with keys: `url`, `status_code`, `headers`, `content`. Useful for debugging HTTP responses, testing server behavior, or inspecting the raw output before applying any parsing.

**Args:**
- `url` — The URL to fetch (must start with http:// or https://).
- `timeout` — Maximum seconds to wait for the HTTP response.

**Returns:** Dict with status_code, headers, and content.

### `fetch_text(url: str, timeout: int = 15) -> str`

Fetch text content from a web page using HTTP and HTML parsing.

Uses httpx to retrieve the page HTML, then BeautifulSoup with lxml to extract clean text, stripping scripts, styles, and navigation elements. Also returns discovered links.

This is the fastest option and works well for most static HTML sites. It does NOT execute JavaScript, so single-page apps (React, Vue, etc.) may return empty or minimal content.

**Args:**
- `url` — The URL to fetch.
- `timeout` — Maximum seconds to wait for the HTTP response.

**Returns:** A string with the extracted text content, followed by a section listing all discovered links.

### `render_page(url: str, timeout: int = 30) -> str`

Render a web page with a headless Chrome browser and extract text.

Spawns `google-chrome` (or `chromium-browser`) in headless mode with `--dump-dom` to execute JavaScript and capture the fully rendered DOM. BeautifulSoup then extracts clean text and links.

This is slower than `fetch_text` but works for JavaScript-rendered pages (SPAs, dynamically loaded content).

**Args:**
- `url` — The URL to render.
- `timeout` — Maximum seconds for Chrome to load the page.

**Returns:** A string with the extracted text content, followed by a section listing all discovered links.

### `take_screenshot(url: str, output_file: str | None = None, timeout: int = 30) -> str`

Capture a screenshot of a web page using headless Chrome.

Spawns Chrome in headless mode with `--headless=new --screenshot` and saves the PNG to a file. Returns the path to the saved screenshot, which can then be read by another agentic object with image capabilities.

**Args:**
- `url` — The URL to screenshot.
- `output_file` — Optional output path. If not provided, a file is created in a temporary directory.
- `timeout` — Maximum seconds for Chrome to load the page.

**Returns:** A string with the path to the saved screenshot, or an error message.

## Usage

To use the WebNavigator, derive from it as a parent class of your agentic object:

```python
class MyAgent(WebNavigator, AgenticObject):
    """You are an agent with web navigation capabilities."""
```

The agent will have access to the `fetch_raw`, `fetch_text`, `render_page`, and `take_screenshot` tools through its system prompt.
