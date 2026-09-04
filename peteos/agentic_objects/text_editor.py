"""TextEditor agentic object — line-based text editing with mtime-based safety."""

from __future__ import annotations

import os

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import agentic_object, tool


@agentic_object()
class TextEditor(AgenticObject):
    """You are a text editor. You hold text as a list of lines in memory.

    Use `load` to read a file into your internal representation.
    Use `read` to view the current text content (refreshes mtime).
    Use `write` to replace all lines with new text (does not write to disk).
    Use `edit` to replace exact line text with new line text (does not write to disk).
    Use `store` to write the lines back to the file (safe against external changes).
    Use `diff` to compare your internal lines against the file on disk.
    Use `clear` to clear your internal lines.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._file_path: str | None = None
        self._file_mtime: float | None = None
        self._trailing_newline: bool = False

    @tool(description="Try to load a file from disk into the editor. Stores the file path and modification time for safe writing.")
    def load(self, file_path: str) -> str:
        """Load a file into the editor's internal line representation."""
        abs_path = os.path.abspath(file_path)
        self._file_path = abs_path
        if not os.path.isfile(abs_path):
            with open(abs_path, "w", encoding="utf-8") as f:
                pass
            self._lines = []
            self._file_mtime = os.path.getmtime(abs_path)
            self._trailing_newline = False
            return f"File not found. Created empty file: {abs_path}."
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._lines = content.splitlines()
            self._file_mtime = os.path.getmtime(abs_path)
            self._trailing_newline = content.endswith("\n")
            return f"Loaded {abs_path} ({len(self._lines)} lines)."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool(description="Return the current text content as a string, joined from the internal lines array. Updates the stored mtime to allow subsequent store().")
    def read(self) -> str:
        """Return the full text content of the editor's lines and refresh the mtime."""
        if not self._file_path:
            return "Error: no file loaded. Use load() first."
        if os.path.isfile(self._file_path):
            self._file_mtime = os.path.getmtime(self._file_path)
        result = "\n".join(self._lines)
        if self._trailing_newline:
            result += "\n"
        return result

    @tool(description="Replace the internal lines array with lines split from the given text. Does not write to disk.")
    def write(self, text: str) -> str:
        """Split text into lines and replace the internal representation."""
        self._lines = text.splitlines()
        self._trailing_newline = text.endswith("\n")
        return f"Replaced ({len(self._lines)} lines)."

    @tool(description="Write the internal lines array back to the file. Fails if the file was modified externally since last read().")
    def store(self) -> str:
        """Write lines to disk, protecting against external file modifications via mtime check."""
        if not self._file_path:
            return "Error: no file loaded. Use load() first."
        try:
            current_mtime = os.path.getmtime(self._file_path)
        except FileNotFoundError:
            return f"Error: file '{self._file_path}' no longer exists."
        if self._file_mtime is None:
            return "Error: internal mtime missing — could not verify file safety."
        if current_mtime != self._file_mtime:
            return (
                f"Error: file '{self._file_path}' was modified externally. "
                f"Your loaded mtime: {self._file_mtime}, current mtime: {current_mtime}. "
                "Use load() to refresh before storing."
            )
        try:
            text = "\n".join(self._lines)
            if self._trailing_newline:
                text += "\n"
            with open(self._file_path, "w", encoding="utf-8") as f:
                f.write(text)
            self._file_mtime = current_mtime
            return f"Stored {len(self._lines)} lines to {self._file_path}."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool(description="Replace old_string with new_string in the text content. Does not write to disk. Both old_string and new_string can span multiple lines. Like the Edit tool — find old_string, replace with new_string, optional replace_all for replacing all occurrences.")
    def edit(self, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace text content with new text, rebuild lines array."""
        if not self._file_path:
            return "Error: no file loaded. Use load() first."
        content = self.read()
        # Count occurrences
        if not replace_all:
            if content.count(old_string) > 1:
                return (
                    f"Error: '{old_string}' found multiple times. "
                    "Set replace_all=True to replace all occurrences."
                )
            if old_string not in content:
                return f"Error: '{old_string}' not found in text."
            new_content = content.replace(old_string, new_string, 1)
        else:
            if old_string not in content:
                return f"Error: '{old_string}' not found in text."
            new_content = content.replace(old_string, new_string)
        # Rebuild lines array from new content
        self._lines = new_content.splitlines()
        self._trailing_newline = new_content.endswith("\n")
        if replace_all:
            count = content.count(old_string)
            return f"Replaced {count} occurrence(s) of '{old_string}'."
        return f"Replaced '{old_string}' with '{new_string}'."

    @tool(description="Compare the internal lines array against the file on disk. Returns a unified-style diff of added, removed, and unchanged lines.")
    def diff(self) -> str:
        """Show the difference between internal lines and the file on disk."""
        if not self._file_path:
            return "Error: no file loaded. Use load() first."
        if not os.path.isfile(self._file_path):
            return f"Error: file no longer exists: {self._file_path}"
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                disk_lines = f.read().splitlines()
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"
        if self._lines == disk_lines:
            return "No differences — internal lines match the file on disk."
        import difflib
        diff = difflib.unified_diff(
            disk_lines, self._lines,
            fromfile=self._file_path, tofile=self._file_path,
            lineterm="",
        )
        return "\n".join(diff)

    @tool(description="Clear the internal lines array and reset file path and mtime tracking.")
    def clear(self) -> None:
        """Clear the editor state."""
        self._lines = []
        self._file_path = None
        self._file_mtime = None
        self._trailing_newline = False
