"""OAP integration tests for TextEditor agentic object."""

from __future__ import annotations

import os
import tempfile

import pytest

from peteos.agentic_objects.text_editor import TextEditor


class TestTextEditorOAP:
    """Integration tests exercising TextEditor tools end-to-end."""

    def test_load_tool(self):
        """Test the load tool reads a file into lines."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            tmp = f.name
        try:
            result = obj.load(tmp)
            assert "Loaded" in result
            assert obj._lines == ["line1", "line2", "line3"]
            assert obj._file_path is not None
            assert obj._file_mtime is not None
        finally:
            os.unlink(tmp)

    def test_read_tool(self):
        """Test the read tool returns joined content."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("alpha\nbeta\ngamma\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.read()
            assert result == "alpha\nbeta\ngamma\n"
        finally:
            os.unlink(tmp)

    def test_write_tool(self):
        """Test the write tool replaces lines."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("keep\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.write("replaced\nlines\n")
            assert "Lines replaced" in result
            assert obj._lines == ["replaced", "lines"]
            # File on disk should be unchanged
            with open(tmp, "r") as f:
                assert f.read() == "keep\n"
        finally:
            os.unlink(tmp)

    def test_edit_tool_single(self):
        """Test the edit tool replaces a single matching line."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("A\nB\nC\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("B", "X")
            assert "Replaced" in result
            assert obj._lines == ["A", "X", "C"]
        finally:
            os.unlink(tmp)

    def test_edit_tool_multi_no_replace_all(self):
        """Test edit tool blocks multiple matches without replace_all."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("X\nY\nX\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("X", "Z")
            assert "multiple times" in result
            assert obj._lines == ["X", "Y", "X"]
        finally:
            os.unlink(tmp)

    def test_edit_tool_multi_replace_all(self):
        """Test edit tool replaces all matches with replace_all=True."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("X\nY\nX\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("X", "Z", replace_all=True)
            assert "Replaced" in result
            assert obj._lines == ["Z", "Y", "Z"]
        finally:
            os.unlink(tmp)

    def test_store_tool_success(self):
        """Test store tool writes lines to disk."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("original\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj._lines = ["modified"]
            obj._trailing_newline = True
            result = obj.store()
            assert "Stored" in result
            with open(tmp, "r") as f:
                assert f.read() == "modified\n"
        finally:
            os.unlink(tmp)

    def test_store_tool_external_modification(self):
        """Test store tool rejects when file was externally modified."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("original\n")
            tmp = f.name
        try:
            obj.load(tmp)
            os.utime(tmp, (os.path.getatime(tmp), os.path.getmtime(tmp) + 1))
            result = obj.store()
            assert "modified externally" in result
        finally:
            os.unlink(tmp)

    def test_diff_tool_no_changes(self):
        """Test diff tool reports no differences."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("same\nlines\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.diff()
            assert "No differences" in result
        finally:
            os.unlink(tmp)

    def test_diff_tool_changes(self):
        """Test diff tool shows differences."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("before\nline two\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj._lines = ["after\nline two\n"]
            result = obj.diff()
            assert "unchanged lines match" not in result.lower()
        finally:
            os.unlink(tmp)

    def test_clear_tool(self):
        """Test clear tool resets all state."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("content\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj.clear()
            assert obj._lines == []
            assert obj._file_path is None
            assert obj._file_mtime is None
        finally:
            os.unlink(tmp)

    def test_workflow_load_edit_store(self):
        """Full workflow: load → edit → store → verify."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("hello\nworld\nfoo\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj.edit("world", "moon")
            obj.store()
            with open(tmp, "r") as f:
                assert f.read() == "hello\nmoon\nfoo\n"
        finally:
            os.unlink(tmp)

    def test_workflow_load_write_store(self):
        """Full workflow: load → write new content → store → verify."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("old\nold\nold\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj.write("new\nnew\nnew\n")
            obj.store()
            with open(tmp, "r") as f:
                assert f.read() == "new\nnew\nnew\n"
        finally:
            os.unlink(tmp)
