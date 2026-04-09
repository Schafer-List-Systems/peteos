"""Utility functions for merging delta events into JSON structures.

This module provides functions for accumulating incremental delta events
into a target data structure. The protocol is provider-agnostic and can
be used to accumulate data from any streaming API.

Key concepts:
- Delta events are incremental updates to a data structure
- Arrays use 'index' fields as control metadata for merge position
- Strings are concatenated for token-by-token accumulation
- Numeric values raise errors (fail-closed design)
"""

from typing import Any, Dict


def merge_delta_into_target(target: Dict, delta: Any, path: str = "") -> None:
    """
    Recursively merge delta event into target structure. Fail-closed.

    This function merges incremental delta events into an accumulated
    target structure. It uses index fields in arrays to determine where
    to merge, and concatenates strings for token-by-token accumulation.

    Control metadata handling:
    - Array items with 'index' field are control metadata
    - The 'index' field itself is NOT accumulated into target
    - Other fields in the control metadata object ARE accumulated

    Fail-closed behavior:
    - Arrays WITHOUT 'index' field: RAISE ValueError
    - Numeric values (int/float): RAISE ValueError (unknown handling)
    - Standalone scalars: RAISE ValueError (should only appear in dict context)

    Args:
        target: Accumulated target dictionary to merge into
        delta: Delta event to merge
        path: Current path for error messages

    Raises:
        ValueError: If delta has unknown patterns (missing index, numbers, etc.)

    Example:
        >>> target = {}
        >>> merge_delta_into_target(target, {
        ...     'tool_calls': [{'index': 0, 'function': {'name': 'calc'}}]
        ... })
        >>> merge_delta_into_target(target, {
        ...     'tool_calls': [{'index': 0, 'function': {'arguments': '{'}}]
        ... })
        >>> target['tool_calls'][0] == {'function': {'name': 'calc', 'arguments': '{'}}
        True
    """
    if isinstance(delta, list):
        for delta_item in delta:
            # Each array item MUST have 'index' field (control metadata)
            if not isinstance(delta_item, dict) or "index" not in delta_item:
                raise ValueError(f"Array item missing 'index' field at path {path}")
            idx = delta_item["index"]
            # Ensure target has enough items
            while len(target) <= idx:
                target.append({})
            # Extract all fields except 'index' and merge into target[idx]
            for key, value in delta_item.items():
                if key == "index":
                    continue  # Index is control metadata, not part of accumulated data
                current_path = f"{path}[{idx}].{key}" if path else f"[{idx}].{key}"
                # If both target[idx][key] and value are dict/list, recurse
                if key in target[idx] and isinstance(target[idx][key], (dict, list)) and isinstance(value, (dict, list)):
                    merge_delta_into_target(target[idx][key], value, current_path)
                # String: concatenate for token accumulation
                elif isinstance(value, str):
                    target[idx][key] = target[idx].get(key, "") + value
                # Numbers: raise error (unknown handling)
                elif isinstance(value, (int, float)):
                    raise ValueError(f"Numeric field not handled at path {current_path}")
                # Otherwise: direct overwrite
                else:
                    target[idx][key] = value

    elif isinstance(delta, dict):
        for key, value in delta.items():
            current_path = f"{path}.{key}" if path else key
            # If both target[key] and delta[key] are dict/list, recurse
            if key in target and isinstance(target[key], (dict, list)) and isinstance(value, (dict, list)):
                merge_delta_into_target(target[key], value, current_path)
            # String: concatenate for token accumulation
            elif isinstance(value, str):
                target[key] = target.get(key, "") + value
            # Numbers: raise error (unknown handling)
            elif isinstance(value, (int, float)):
                raise ValueError(f"Numeric field not handled at path {current_path}")
            # Otherwise: direct overwrite
            else:
                target[key] = value

    else:
        # Standalone scalar should not happen in valid delta events
        raise ValueError(f"Standalone scalar at path {path}: {delta!r}")


def translate_delta_event(event: Dict, translations: Dict[str, str]) -> Dict:
    """
    Translate SSE delta event to uniform format, preserving all index fields.

    For each translation "source_path" -> "target_key":
    1. Parse source path to identify array indices
    2. Extract value at that path
    3. Propagate index values to nested arrays
    4. Store under target_key in uniform format

    Supports type discriminator paths: "content_block_start.content_block.text"
    will match event["type"] == "content_block_start" then extract content_block.text

    Args:
        event: Raw SSE event (e.g., from OpenAI or Anthropic API)
        translations: Dict mapping source path -> target key

    Returns:
        Translated event with uniform keys and preserved index fields

    Example:
        >>> event = {'choices': [{'index': 0, 'delta': {
        ...     'tool_calls': [{'index': 0, 'function': {'name': 'calc'}}]
        ... }}]}
        >>> translations = {'choices[*].delta.tool_calls': 'tool_calls'}
        >>> translate_delta_event(event, translations)
        {'tool_calls': [{'index': 0, 'function': {'name': 'calc'}}]}
    """
    from .dict_path import extract_with_indices, get_value_at_path

    result = {}

    for source_path, target_key in translations.items():
        # Parse path: "choices[*].delta.tool_calls" -> ["choices", "delta", "tool_calls"]
        path_parts = source_path.split(".")

        # First try extract_with_indices (for paths like "choices[*].delta.tool_calls")
        translated = extract_with_indices(event, path_parts)

        # If that fails and path starts with event type, try type discriminator
        # This handles paths like "content_block_start.content_block.text"
        if translated is None:
            first_part = path_parts[0]
            # If first part matches event["type"], use type discriminator
            if event.get("type") == first_part and len(path_parts) > 1:
                # Use remaining path with type discriminator
                remaining_path = ".".join(path_parts[1:])
                translated = get_value_at_path(event, remaining_path, type_discriminator=True)

        if translated is not None:
            result[target_key] = translated

    return result
