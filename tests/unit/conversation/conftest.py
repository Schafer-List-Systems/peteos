"""Prevents peteos/__init__.py from executing so conversation submodules can import.

Creates a minimal ``peteos`` module stub using ``types.ModuleType`` with a
proper ``__path__`` so importlib can still discover real submodules (like
peteos.conversation, peteos.logger) from the filesystem, but never loads the
broken peteos/__init__.py.
"""

import logging
import os
import sys
import types

_this_dir = os.path.dirname(__file__)
_peteos_root = os.path.normpath(os.path.join(_this_dir, "..", "..", "..", "peteos"))

_peeteos = types.ModuleType("peteos")
_peeteos.__path__ = [_peteos_root]
_peeteos.__file__ = "<stub>"
_peeteos.__package__ = "peteos"
sys.modules["peteos"] = _peeteos

_logger = types.ModuleType("peteos.logger")
_logger.get_logger = logging.getLogger
sys.modules["peteos.logger"] = _logger

# Import message subclasses so they register with MessageRegistry.
# These must be imported AFTER the stub replaces sys.modules["peteos"].
import peteos.conversation.system_prompt_message  # noqa: F401
import peteos.conversation.tool_definitions_message  # noqa: F401
