"""Utility functions for peteos."""


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
