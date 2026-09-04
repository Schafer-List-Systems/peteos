"""JSON module selection with graceful fallback chain.

Import order: json5 → jsonc → stdlib json.

Wraps json5/jsonc loaders so their ValueError is surfaced as
json.JSONDecodeError — keeping the exception interface stable
regardless of which backend is in use.
"""

import json as _strict

try:
    import json5 as _json  # type: ignore[import-untyped]
except ImportError:
    try:
        import jsonc as _json  # type: ignore[import-untyped]
    except ImportError:
        _json = _strict

JSONDecodeError = _strict.JSONDecodeError


def loads(s: str, **kwargs) -> object:
    try:
        return _json.loads(s, **kwargs)
    except ValueError as e:
        raise JSONDecodeError(str(e), "", 0) from e


def load(fp, **kwargs) -> object:
    try:
        return _json.load(fp, **kwargs)
    except ValueError as e:
        raise JSONDecodeError(str(e), "", 0) from e


def dumps(obj, **kwargs) -> str:
    return _strict.dumps(obj, **kwargs)


def dump(obj, fp, **kwargs) -> None:
    _strict.dump(obj, fp, **kwargs)
