# TextEditor

A line-based text editor agentic object with mtime-based safety for file writes.

## Description

`TextEditor` is an agentic object that holds text as a list of lines in memory. It provides tools to load files into the editor, view and modify content, and safely write changes back to disk. The mtime-based safety check prevents accidentally overwriting files that have been modified externally.

The editor is useful whenever the agent needs to create, inspect, or modify text files — source code, configuration files, data files, or any line-based text.

## Tools

### `load(file_path: str) -> str`

Load a file from disk into the editor's internal line representation. Stores the absolute file path and modification time.

**Args:**
- `file_path` — The path to the file to load.

**Returns:** A message with the number of lines loaded, or an error if the file is not found.

### `read() -> str`

Return the current text content as a string, joined from the internal lines array. Updates the stored modification time to allow subsequent `store()`.

**Returns:** The full text content as a string, or an error if no file has been loaded.

### `write(text: str) -> str`

Replace the internal lines array with lines split from the given text. Does not write to disk.

**Args:**
- `text` — The new text content.

**Returns:** A message with the number of lines replaced.

### `edit(old_string: str, new_string: str, replace_all: bool = False) -> str`

Replace `old_string` with `new_string` in the text content. Does not write to disk. Both arguments can span multiple lines — a multi-line block can be replaced with any number of lines. Works like the Edit tool but operates on the editor's in-memory content.

**Args:**
- `old_string` — The exact text to find (may contain newlines).
- `new_string` — The replacement text (may contain newlines).
- `replace_all` — If True, replace all occurrences. Default is False.

**Returns:** A message with the result, or an error if `old_string` is not found or multiple occurrences exist without `replace_all=True`.

### `store() -> str`

Write the internal lines array back to the file on disk. Checks the file's modification time against the stored value and fails if the file was modified externally since the last `load()` or `read()`.

**Returns:** A message with the number of lines stored, or an error if the file was externally modified or does not exist.

### `diff() -> str`

Compare the internal lines array against the file on disk. Returns a unified-style diff showing added, removed, and unchanged lines.

**Returns:** A unified diff string, or "No differences" if the lines match the file.

### `clear() -> None`

Clear the internal lines array and reset file path and modification time tracking.

## Usage

To use the TextEditor, derive from it as a parent class of your agentic object:

```python
from peteos.oap.base import AgenticObjectBase

class MyAgent(TextEditor, AgenticObjectBase):
    """You are an agent that can edit text files."""
```

The agent will have access to the `load`, `read`, `write`, `edit`, `store`, `diff`, and `clear` tools through its system prompt.

## Typical Workflow

```
1. load("/path/to/file.txt")   — read file into lines, capture mtime
2. read()                       — view current content (refreshes mtime)
3. edit("old line", "new line") — modify specific lines in memory
4. diff()                       — check what changed vs. disk
5. store()                      — write changes back (safety check runs)
```

The safety check in `store()` ensures that if another process modified the file between `load()` and `store()`, the write is rejected and the agent must use `load()` to refresh its state first.
