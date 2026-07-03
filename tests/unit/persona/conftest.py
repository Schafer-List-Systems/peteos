"""Prevent peteos/__init__.py from executing during persona tests."""

import pytest
import sys
import types

_this_dir = __import__("os").path.dirname(__file__)
_peteos_root = __import__("os").path.normpath(
    __import__("os").path.join(_this_dir, "..", "..", "..", "peteos")
)

_peeteos = types.ModuleType("peteos")
_peeteos.__path__ = [_peteos_root]
_peeteos.__file__ = "<stub>"
_peeteos.__package__ = "peteos"
sys.modules["peteos"] = _peeteos

_logger = types.ModuleType("peteos.logger")
_logger.get_logger = __import__("logging").getLogger
sys.modules["peteos.logger"] = _logger


@pytest.fixture(autouse=True)
def _clear_rolemanager():
    """Clear RoleManager class-level state to prevent test pollution."""
    from peteos.persona.rolemanager import RoleManager
    RoleManager._roles.clear()
    yield
    RoleManager._roles.clear()
