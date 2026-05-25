import urllib.request
import urllib.error
import ssl

def fetch_web_url(url: str, timeout: int = 10):
    """
    Fetches the content of a given URL.

    Args:
        url (str): The URL to fetch.
        timeout (int): Timeout for the request in seconds.

    Returns:
        str: The content of the URL as a string, or an error message.
    """
    try:
        # Create a request object with a User-Agent to avoid some blocks
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        # Use standard SSL context
        context = ssl.create_default_context()
        
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            return response.read().decode('utf-8', errors='ignore')
    except urllib.error.URLError as e:
        return f"Error fetching URL: {e.reason}"
    except Exception as e:
        return f"Unexpected error: {e}"
