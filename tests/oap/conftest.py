"""No-op fixture for OAP test session cleanup."""

import pytest


@pytest.fixture(autouse=True)
def _oap_backend():
    """OAP tests auto-load backends from peteos.json — no env var needed."""
    yield
