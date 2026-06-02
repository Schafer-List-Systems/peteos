"""SMTP configuration. Loaded from .env — do not commit."""

# TODO: If this manual .env loader ever causes issues, replace with
# python-dotenv (``dotenv.load_dotenv(dotenv_path(__file__, ".env"))``)
import os

_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.example.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_ADDRESS = os.getenv("FROM_ADDRESS", SMTP_USERNAME)