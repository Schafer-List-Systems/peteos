# Uniform Delta Protocol for JSON Structure Accumulation

## Uniform Response Format

This section defines the **target uniform format** that all providers translate streaming responses into after accumulation. This is the structure of a complete response after processing all SSE deltas, or in non-streaming mode the direct translation of the API response.

### Complete Response (After Stream Accumulation)

All content (thinking, tool calls, text) is stored in a single `content` array to support interleaved blocks:

```python
result = {
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "content": "reasoning/thinking text"
        },
        {
            "type": "tool_use",
            "id": "toolu_xxx",
            "name": "function_name",
            "arguments": '{"key": "value"}'
        },
        {
            "type": "text",
            "content": "final response text"
        }
    ],
    "stop_reason": "end_turn"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `role` | string | Always "assistant" for model responses |
| `content` | array | Array of content blocks (thinking, tool_use, text) |
| `stop_reason` | string | Final stop reason from message_delta or equivalent |

### Content Block Types

| Type | Fields | Description |
|------|--------|-------------|
| `thinking` | `content` string | Reasoning/thoughts before tool calls or final answer |
| `tool_use` | `id`, `name`, `arguments` | Tool/function call with JSON arguments |
| `text` | `content` string | Final text response to user |

### Example: Interleaved Content

A single response may have multiple content blocks in any order:

```python
result = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "content": "Let me calculate..."},
        {"type": "tool_use", "id": "t1", "name": "add", "arguments": '{"a": 1, "b": 2}'},
        {"type": "thinking", "content": "Now let me format the answer"},
        {"type": "text", "content": "The sum is 3"}
    ],
    "stop_reason": "end_turn"
}
```

**Note:** During streaming, array items have an `index` field for delta merge control. This field is stripped from the final accumulated result.

### Delta Structure Format

During streaming, the target uniform format is updated incrementally using delta events. The delta structures follow these rules:

#### Array Items

Each item in an array MUST have an `index` field for merge control:

```python
{
    "role": "assistant",
    "content": [
        {
            "index": 0,           # Control: position in array
            "type": "thinking",
            "content": "text fragment"
        }
    ],
    "stop_reason": None
}
```

- The `index` field is control metadata (NOT accumulated into final result)
- The `index` value equals the array position
- Array items are padded to `index + 1` if needed

#### String Values

String fields are accumulated by concatenation:

```python
# First delta
{"content": [{"index": 0, "type": "thinking", "content": "Let me"}]}

# Second delta
{"content": [{"index": 0, "type": "thinking", "content": " think..."}]}

# After accumulation
{"content": [{"type": "thinking", "content": "Let me think..."}]}
```

- Each delta contains a fragment
- Fragments are concatenated in order
- Final result has no `index` field

#### Nested Objects

Nested objects in array items are merged recursively:

```python
# First delta: tool call name
{"content": [{"index": 0, "type": "tool_use", "name": "add"}]}

# Second delta: tool call arguments
{"content": [{"index": 0, "arguments": '{"a": 1}'}]}

# After accumulation
{"content": [{"type": "tool_use", "name": "add", "arguments": '{"a": 1}'}]}
```

#### Bijective Transformation

The transformation between delta and complete structure is bijective:

1. **Delta → Complete**: Merge deltas in order, strip `index` fields from final result
2. **Complete → Delta**: Each final item can be reconstructed by finding its `index` from array position

```python
# Delta structure (during streaming)
delta = {"content": [{"index": 0, "type": "text", "content": "Hello"}]}

# Complete structure (after accumulation)
complete = {"content": [{"type": "text", "content": "Hello"}]}

# Bijective mapping: item[0] in complete ↔ item with index:0 in delta
```

**Rules:**
- `index` in delta → array position in complete
- All other fields → same field in complete (concatenated if string, merged if object)

---

## Overview

This document describes a general-purpose protocol for merging and accumulating JSON data structures over incremental delta streams. The protocol is provider-agnostic and can be used to accumulate data from any streaming API that produces incremental updates.

The implementation is used by the chatbot system to accumulate responses from different LLM providers (OpenAI, Anthropic, etc.) into a unified data structure.

## Core Concepts

### Delta Events

A **delta event** is an incremental update that represents a partial change to a data structure. Delta events are characterized by:

1. **Incremental updates**: Each event represents a small piece of the final data
2. **Position indicators**: Arrays use `index` fields to indicate where updates should be applied
3. **String concatenation**: String values are fragments that need to be joined together

### Uniform Delta Format

The **uniform delta format** is a standardized representation of delta events that allows the merge function to work generically across different data sources.

### Index Field Semantics

The `index` field has a specific role:
- **Purpose**: Controls WHERE in the target array to merge the delta
- **Lifetime**: Used ONLY during merge operation
- **Result**: NOT kept in the final accumulated data (redundant metadata)

Example:
```python
# Delta during merge
{
    "content": [{
        "index": 0,      # Control: merge at content[0]
        "type": "text",
        "content": "Hello"
    }]
}

# After merge, result does NOT include index
{
    "content": [{
        "type": "text",    # index removed - data only
        "content": "Hello"
    }]
}
```

The `index` is stripped from accumulated results because:
1. Array position already encodes the order
2. Keeping it would be redundant metadata
3. Different providers may have different index semantics

## The Protocol

### Step 1: Translation

Each data source (API, protocol, format) defines a **translation function** that converts native delta events into the uniform delta format.

```
Native Delta Event → [Translation Function] → Uniform Delta Event
```

The translation function is responsible for:
- Extracting relevant data fields from the native format
- Converting nested structures to the uniform format
- Preserving position information (`index` fields)

### Step 2: Merging

The **merge function** accumulates uniform delta events into a target data structure.

```python
merge_delta_into_target(target: Dict, delta: Any, path: str = "") -> None
```

The merge function applies the following rules:

#### String Values

Strings are accumulated by concatenation:

```python
# Before
target = {"message": "Hello"}
delta = {"message": " World"}

# After merge
target = {"message": "Hello World"}
```

#### Arrays with Index

Arrays use the `index` field to determine where to merge. The `index` is used for positioning but NOT included in the final result:

```python
# Before
target = {"tool_calls": []}
delta = {
    "tool_calls": [{
        "index": 0,        # Used to position at tool_calls[0]
        "function": {"name": "calculate"}
    }]
}

# After merge (index stripped)
target = {
    "tool_calls": [{
        "function": {"name": "calculate"}  # index removed
    }]
}

# Subsequent delta
delta = {
    "tool_calls": [{
        "index": 0,
        "function": {"arguments": '{"expr": "2+2"}'}
    }]
}

# After second merge
target = {
    "tool_calls": [{
        "function": {
            "name": "calculate",
            "arguments": '{"expr": "2+2"}'
        }
    }]
}
```

#### Objects

Nested objects are merged recursively.

#### Numeric Values

Numeric values (int, float) are **not supported** in delta events. If encountered, the merge function raises an error. This is a fail-closed design decision to prevent unexpected behavior.

### Step 3: Accumulated Result

After processing all delta events, the target structure contains the complete accumulated data:

```python
result = {
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "content": "Let me think about this..."
        },
        {
            "type": "tool_use",
            "id": "toolu_xxx",
            "name": "calculate",
            "arguments": '{"expression": "2+2"}'
        }
    ],
    "stop_reason": "tool_use"
}
```

## Design Principles

### 1. Fail-Closed Behavior

The merge function raises errors for unexpected patterns rather than guessing:

- Arrays without `index` fields → Error
- Numeric values → Error
- Unknown data patterns → Error

This prevents silent corruption of data structures.

### 2. Control Metadata Pattern

Arrays use the `index` field as **control metadata**. The `index` value:
- Tells the merge function which array position to update
- Is NOT accumulated into the final result (except when explicitly part of the data)

Example:
```python
# The 'index' field controls WHERE to merge
delta = {
    "tool_calls": [{
        "index": 0,        # Merge into position 0
        "function": {...}  # Actual data to accumulate
    }]
}

# The result does not include the index field as data
result = {
    "tool_calls": [{
        "function": {...}  # index is metadata, not data
    }]
}
```

### 3. Position Pointer Pattern

Some protocols (like Anthropic) use `index` as a **position pointer** at the event level, not inside arrays:

```python
# Anthropic format
event = {
    "type": "content_block_delta",
    "delta": {"thinking": "Hello"},
    "index": 0  # Point to content[0]
}

# Translation converts this to uniform format
uniform_delta = {
    "content": [{
        "index": 0,  # Now inside the array item
        "type": "thinking",
        "content": "Hello"
    }]
}
```

### 4. Provider-Specific Translation

Each data source implements its own translation function. The merge function is provider-agnostic and only works with uniform format.

```python
class ChatBotResponse:
    def _translate_event(self, native_event) -> UniformDelta:
        # Override in subclasses
        pass
    
    def _accumulate_event(self, delta) -> None:
        merge_delta_into_target(self._data, delta)
```

## Implementation

### Core Function

```python
def merge_delta_into_target(target: dict, delta: any, path: str = "") -> None:
    """
    Recursively merge delta event into target structure. Fail-closed.
    
    Args:
        target: Accumulated target dictionary to merge into
        delta: Delta event to merge
        path: Current path for error messages
    
    Raises:
        ValueError: If delta has unknown patterns (missing index, numbers, etc.)
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
                    continue
                # Recurse or accumulate based on value type
                if isinstance(value, (dict, list)):
                    merge_delta_into_target(target[idx][key], value, f"{path}[{idx}].{key}")
                elif isinstance(value, str):
                    target[idx][key] = target[idx].get(key, "") + value
                elif isinstance(value, (int, float)):
                    raise ValueError(f"Numeric field not handled at path {path}")
                else:
                    target[idx][key] = value
    
    elif isinstance(delta, dict):
        for key, value in delta.items():
            current_path = f"{path}.{key}" if path else key
            if key in target and isinstance(target[key], (dict, list)) and isinstance(value, (dict, list)):
                merge_delta_into_target(target[key], value, current_path)
            elif isinstance(value, str):
                target[key] = target.get(key, "") + value
            elif isinstance(value, (int, float)):
                raise ValueError(f"Numeric field not handled at path {current_path}")
            else:
                target[key] = value
    
    else:
        raise ValueError(f"Standalone scalar at path {path}: {delta!r}")
```

## Example: OpenAI Stream

**Native OpenAI SSE Event:**
```json
{
  "choices": [{
    "delta": {
      "reasoning": "The",
      "content": "Hello",
      "tool_calls": [{
        "index": 0,
        "function": {"name": "calc"}
      }]
    }
  }]
}
```

**Translation to Uniform:**
```python
{
    "content": [{
        "index": 0,
        "type": "thinking",
        "content": "The"
    }, {
        "index": 1,
        "type": "text",
        "content": "Hello"
    }, {
        "index": 2,
        "type": "tool_use",
        "name": "calc"
    }]
}
```

**After Accumulation:**
```python
{
    "content": [{
        "type": "thinking",
        "content": "The user wants me to calculate..."
    }, {
        "type": "text",
        "content": "Hello"
    }, {
        "type": "tool_use",
        "name": "calc",
        "arguments": '{"expr": "2+2"}'
    }]
}
```

## Example: Anthropic Stream

**Native Anthropic SSE Event:**
```json
{
  "type": "content_block_delta",
  "delta": {"type": "thinking_delta", "thinking": "Hello"},
  "index": 0
}
```

**Translation to Uniform:**
```python
{
    "content": [{
        "index": 0,
        "type": "thinking",
        "content": "Hello"
    }]
}
```

**After Accumulation:**
```python
{
    "content": [{
        "type": "thinking",
        "content": "Hello world..."
    }]
}
```

## Benefits

1. **Provider Independence**: Add new providers by implementing translation only
2. **Testability**: Merge function can be tested with controlled delta events
3. **Fail-Closed**: Errors are raised early for unexpected patterns
4. **Extensible**: New data patterns can be added to merge function

## Related Documentation

### Provider Translation Specifications

- [OpenAI Delta Translation](openai-delta-translation.md) - How OpenAI SSE events translate to uniform format
- [Anthropic Delta Translation](anthropic-delta-translation.md) - How Anthropic SSE events translate to uniform format

### Core Protocol

- [Uniform Delta Protocol](uniform-delta-protocol.md) - Target format and merge protocol

### Related Documentation

- [Chatbot System Architecture](chatbot.md)
- [OpenAI API Integration](../integrations/openai.md)
- [Anthropic API Integration](../integrations/anthropic.md)
