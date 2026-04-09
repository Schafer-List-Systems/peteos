"""Utility functions for accessing data at arbitrary paths in nested structures.

This module provides functions for traversing and extracting data from
nested dictionaries and lists using path notation.
"""


def get_value_at_path(data: dict, path: str, type_discriminator: bool = True) -> any:
    """
    Get value from nested dict using path notation.

    Supports:
    - Simple: "field" -> data["field"]
    - Specific index: "choices[0]" -> data["choices"][0]
    - Wildcard: "choices[*]" -> iterate all choices
    - Type discriminator: "content_block_start.content_block.text"
      When path component doesn't exist as key but matches "type" field,
      uses current dict and continues with remaining path

    Args:
        data: Source dictionary
        path: Path like "choices[*].delta.content" or "content_block_start.content_block.text"
        type_discriminator: If True, allow "type" field matching at first component

    Returns:
        Value at path, or None if not found
    """
    parts = path.split(".")
    current = data

    i = 0
    while i < len(parts):
        part = parts[i]
        is_first_part = (i == 0)

        if "[" in part:
            # Handle array access: "choices[0]" or "choices[*]"
            base, index_str = part.split("[")
            index = index_str.rstrip("]")

            if current is None:
                return None
            if not isinstance(current, dict):
                return None
            if base not in current:
                return None

            if index == "*":
                # Wildcard - return list of values for all items
                array = current[base]
                if not isinstance(array, list):
                    return None
                # Recursively get rest of path for each item
                result = []
                for item in array:
                    val = get_value_at_path(item, ".".join(parts[i + 1:]), type_discriminator=False)
                    if val is not None:
                        result.append(val)
                return result if result else None

            else:
                # Specific index
                try:
                    idx = int(index)
                    array = current[base]
                    if not isinstance(array, list) or idx >= len(array):
                        return None
                    current = array[idx]
                except (ValueError, IndexError):
                    return None

        else:
            # Simple key access with type discriminator fallback
            if current is None:
                return None
            if not isinstance(current, dict):
                return None

            # If part is not a key and this is the first component, check type discriminator
            # This allows paths like "content_block_start.content_block.text"
            # to work with events like {"type": "content_block_start", "content_block": {...}}
            if part not in current:
                if is_first_part and type_discriminator and current.get("type") == part:
                    # Type matches - skip this path component, stay in current dict
                    pass  # Don't change current, just advance i below
                else:
                    return None
            else:
                current = current[part]

        i += 1

    return current


def extract_with_indices(data: dict, path_parts: list, parent_index: int | None = None) -> any:
    """
    Extract value at path, propagating index from parent arrays.

    If parent has 'index': 0, extracted array items will have index field set.

    Args:
        data: Source dictionary
        path_parts: Parsed path components (e.g., ["choices", "delta", "tool_calls"])
        parent_index: Index of parent array item, if any

    Returns:
        Extracted value with index fields propagated, or None if not found
    """
    if not path_parts:
        return propagate_indices(data, parent_index)

    part = path_parts[0]
    remaining = path_parts[1:]
    remaining_str = ".".join(remaining) if remaining else None

    if "[" in part and part.endswith("]"):
        # Array access: "choices[0]" or "choices[*]"
        base = part.split("[")[0]
        index_str = part.split("[")[1].rstrip("]")

        if base not in data or not isinstance(data[base], list):
            return None

        if index_str == "*":
            # Wildcard - extract from first item only (delta events are per-index)
            if data[base]:
                return extract_with_indices(data[base][0], remaining, parent_index=0)
            return None
        else:
            # Specific index like "choices[0]"
            try:
                idx = int(index_str)
                if idx < len(data[base]):
                    return extract_with_indices(data[base][idx], remaining, parent_index=idx)
                return None
            except ValueError:
                return None

    # Simple key access
    if part in data:
        return extract_with_indices(data[part], remaining, parent_index) if remaining else data[part]
    return None


def propagate_indices(obj: any, parent_index: int | None) -> any:
    """
    Ensure all array items have 'index' field matching parent_index.

    This propagates index from parent arrays to their children, which is
    necessary for delta merge to work correctly with nested arrays.

    Args:
        obj: Object to propagate indices into
        parent_index: Index of parent array item

    Returns:
        Object with index fields propagated
    """
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                # Only set index if not already present
                if parent_index is not None and "index" not in item:
                    item["index"] = parent_index
            # Recurse with current item's index
            propagate_indices(item, parent_index if parent_index is not None else i)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            propagate_indices(value, parent_index)
    return obj
