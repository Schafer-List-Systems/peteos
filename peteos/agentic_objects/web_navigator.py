"""WebNavigator - agentic object for web fetching, rendering, and screenshots."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object, tool
from peteos.utils import get_logger

_logger = get_logger(__name__)

if TYPE_CHECKING:
    from peteos.engine import Runner


# Chrome binary to prefer — try Google Chrome first, fall back to Chromium.
def _find_chrome() -> str:
    for binary in ("google-chrome", "chromium-browser", "chromium"):
        try:
            result = subprocess.run(
                ("which", binary),
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split()[0]
        except Exception:
            continue
    return "google-chrome"


_CHROME_BIN = _find_chrome()


def _fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch raw HTML from a URL using httpx."""
    import httpx
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as e:
        return f"Error fetching URL: {type(e).__name__}: {e}"


def _extract_text(html: str) -> tuple[str, list[dict]]:
    """Extract clean text and links from HTML using BeautifulSoup.

    Returns:
        Tuple of (extracted text, list of link dicts with title and href).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, nav, header, footer, noscript, svg
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript", "svg", "iframe", "noscript"]):
        tag.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str)):
        parent = comment.parent
        if parent and parent.name not in ("br", "p", "div", "li", "td", "th", "a", "span"):
            comment.extract()

    # Extract text from body or fallback to full soup
    body = soup.find("body") or soup
    text = body.get_text(separator="\n", strip=True)

    # Extract links
    links = []
    for a in soup.find_all("a", href=True):
        title = (a.get_text(strip=True) or a.get("title", "") or a["href"])[:100]
        links.append({"title": title, "href": a["href"]})

    return text, links


@agentic_object(allow_code_execution=False)
class WebNavigator(AgenticObject):
    """You are a web navigator agentic object with tools to fetch, render, and screenshot web pages.

    Use ``fetch_text`` for fast retrieval of static HTML pages (most informational sites).
    Use ``render_page`` when a page requires JavaScript execution to display content (SPAs like React, Vue).
    Use ``take_screenshot`` when visual context of the page is needed — the screenshot is saved as a file
    that can then be read by another agent with image capabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self._temp_dir: Path | None = None

    @property
    def temp_dir(self) -> Path:
        """Temporary directory for screenshot output."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="web_navigator_"))
        return self._temp_dir

    @tool
    def fetch_raw(self, url: str, timeout: int = 15) -> dict:
        """Fetch the raw HTTP response including headers and body.

        Returns a structured dict with the status code, all response
        headers as a dict, and the raw response body. Useful for debugging,
        testing HTTP responses, or inspecting the server's exact output
        before applying any parsing.

        Args:
            url: The URL to fetch.
            timeout: Maximum seconds to wait for the HTTP response.

        Returns:
            Dict with keys: url, status_code, headers, content.
        """
        if not url.startswith(("http://", "https://")):
            return {"url": url, "status_code": -1, "headers": {}, "content": "Error: URL must start with http:// or https://."}

        try:
            import httpx
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"},
            ) as client:
                response = client.get(url)
                headers = {k: v for k, v in response.headers.items()}
                return {
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "headers": headers,
                    "content": response.text,
                }
        except Exception as e:
            return {"url": url, "status_code": -1, "headers": {}, "content": f"Error: {type(e).__name__}: {e}"}

    @tool
    def fetch_text(self, url: str, timeout: int = 15) -> str:
        """Fetch text content from a web page using HTTP and HTML parsing.

        Uses httpx to retrieve the page HTML, then BeautifulSoup with lxml
        to extract clean text, stripping scripts, styles, and navigation
        elements. Also returns discovered links.

        This is the fastest option and works well for most static HTML sites.
        It does NOT execute JavaScript, so single-page apps (React, Vue, etc.)
        may return empty or minimal content.

        Args:
            url: The URL to fetch.
            timeout: Maximum seconds to wait for the HTTP response.

        Returns:
            A string with the extracted text content, followed by a section
            listing all discovered links with their titles and hrefs.
        """
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        html = _fetch_html(url, timeout=timeout)
        if html.startswith("Error"):
            return html

        text, links = _extract_text(html)

        result = text
        if links:
            result += f"\n\n--- Links ({len(links)} found) ---\n"
            for link in links:
                result += f"- {link['title']}: {link['href']}\n"

        return result

    @tool
    def render_page(self, url: str, timeout: int = 30) -> str:
        """Render a web page with a headless Chrome browser and extract text.

        Spawns ``google-chrome`` (or ``chromium-browser``) in headless mode
        with ``--dump-dom`` to execute JavaScript and capture the fully
        rendered DOM. BeautifulSoup then extracts clean text and links.

        This is slower than ``fetch_text`` but works for JavaScript-rendered
        pages (SPAs, dynamically loaded content).

        Args:
            url: The URL to render.
            timeout: Maximum seconds for Chrome to load the page.

        Returns:
            A string with the extracted text content, followed by a section
            listing all discovered links.
        """
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        try:
            proc = subprocess.run(
                (
                    _CHROME_BIN,
                    "--headless=new",
                    "--no-sandbox",
                    "--dump-dom",
                    "--disable-gpu",
                    f"--timeout-ms={timeout * 1000}",
                    url,
                ),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode != 0:
                error_detail = (proc.stderr or "non-zero exit code").strip()
                return f"Error rendering page: Chrome exited with code {proc.returncode}. {error_detail}"

            html = proc.stdout
            text, links = _extract_text(html)

            result = text
            if links:
                result += f"\n\n--- Links ({len(links)} found) ---\n"
                for link in links:
                    result += f"- {link['title']}: {link['href']}\n"

            return result

        except subprocess.TimeoutExpired:
            return f"Error: Page render timed out after {timeout} seconds."
        except Exception as e:
            return f"Error rendering page: {type(e).__name__}: {e}"

    @tool
    def take_screenshot(self, url: str, output_file: str | None = None, timeout: int = 30) -> str:
        """Capture a screenshot of a web page using headless Chrome.

        Spawns Chrome in headless mode with ``--headless=new --screenshot``
        and saves the PNG to a file. Returns the path to the saved screenshot.

        The screenshot can then be read by another agentic object with image
        capabilities to provide visual context.

        Args:
            url: The URL to screenshot.
            output_file: Optional output path. If not provided, a file is
                created in a temporary directory with a generated name.
            timeout: Maximum seconds for Chrome to load the page.

        Returns:
            A string with the path to the saved screenshot, or an error
            message if the screenshot failed.
        """
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        if output_file is None:
            output_path = self.temp_dir / f"Screenshot_{Path(url).name}.png"
        else:
            output_path = Path(output_file).resolve()
            # Guard against directory traversal
            try:
                output_path.relative_to(self.temp_dir.resolve() if output_file.startswith("..") else output_path.parent)
            except ValueError:
                return f"Error: Output path '{output_file}' is outside the allowed workspace."

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                (
                    _CHROME_BIN,
                    "--headless=new",
                    "--no-sandbox",
                    f"--screenshot={output_path}",
                    "--window-size=1280,800",
                    "--disable-gpu",
                    f"--timeout-ms={timeout * 1000}",
                    url,
                ),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode != 0 or not output_path.is_file():
                error_detail = (proc.stderr or "Chrome exited unexpectedly").strip()
                return f"Error capturing screenshot: {error_detail}"

            return f"Screenshot saved to: {output_path}"

        except subprocess.TimeoutExpired:
            return f"Error: Screenshot timed out after {timeout} seconds."
        except Exception as e:
            return f"Error capturing screenshot: {type(e).__name__}: {e}"
