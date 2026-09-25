"""Result serialization helpers — ensure any value becomes a string for the LLM.

Preserves strings as-is. Serializes dicts/list/bool as proper JSON.
Falls back to repr for non-serializable types.
"""

from __future__ import annotations

import json as _json

from peteos.utils.json import JSONDecodeError


def serialize(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, bool, int, float)) and value is not None:
        try:
            return _json.dumps(value)
        except (TypeError, ValueError):
            pass
    return str(value)
