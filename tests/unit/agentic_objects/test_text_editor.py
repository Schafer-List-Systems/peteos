"""Unit tests for TextEditor agentic object."""

from __future__ import annotations

import os
import tempfile

from peteos.agentic_objects.text_editor import TextEditor


class TestTextEditor:
    """Tests for the TextEditor OAP."""

    def test_load_nonexistent_file_creates_empty_file(self):
        """Test loading a file that does not exist creates an empty file."""
        obj = TextEditor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "newfile.txt")
            result = obj.load(path)
            assert "Created empty file" in result
            assert obj._file_path == os.path.abspath(path)
            assert obj._lines == []

    def test_load_and_read(self):
        """Test loading a file and reading its content."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            tmp = f.name
        try:
            result = obj.load(tmp)
            assert "Loaded" in result
            assert obj._lines == ["line1", "line2", "line3"]
            assert obj._file_path == os.path.abspath(tmp)
            assert obj._file_mtime is not None
            content = obj.read()
            assert content == "line1\nline2\nline3\n"
        finally:
            os.unlink(tmp)

    def test_read_without_load(self):
        """Test read() fails without prior load()."""
        obj = TextEditor()
        result = obj.read()
        assert "no file loaded" in result.lower()

    def test_write_replaces_lines(self):
        """Test write() replaces the internal lines array."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("original\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.write("new line one\nnew line two")
            assert "Replaced" in result
            assert obj._lines == ["new line one", "new line two"]
        finally:
            os.unlink(tmp)

    def test_edit_single_match(self):
        """Test edit() replaces the first matching line."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("apple\nbanana\napple\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("banana", "orange")
            assert "Replaced" in result
            assert obj._lines == ["apple", "orange", "apple"]
        finally:
            os.unlink(tmp)

    def test_edit_no_match(self):
        """Test edit() fails when old_string is not found."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("apple\nbanana\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("cherry", "grape")
            assert "not found" in result
            assert obj._lines == ["apple", "banana"]
        finally:
            os.unlink(tmp)

    def test_edit_multiple_matches_no_replace_all(self):
        """Test edit() fails with multiple matches when replace_all is False."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("apple\nbanana\napple\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("apple", "orange")
            assert "multiple times" in result
            # Lines should be unchanged
            assert obj._lines == ["apple", "banana", "apple"]
        finally:
            os.unlink(tmp)

    def test_edit_multiple_matches_with_replace_all(self):
        """Test edit() replaces all matching lines when replace_all=True."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("apple\nbanana\napple\n")
            tmp = f.name
        try:
            obj.load(tmp)
            result = obj.edit("apple", "orange", replace_all=True)
            assert "Replaced" in result and "occurrence" in result.lower()
            assert obj._lines == ["orange", "banana", "orange"]
        finally:
            os.unlink(tmp)

    def test_edit_without_load(self):
        """Test edit() fails without prior load()."""
        obj = TextEditor()
        result = obj.edit("old", "new")
        assert "no file loaded" in result

    def test_store_basic(self):
        """Test store() writes lines back to disk."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("original\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj._lines = ["replaced"]
            obj._trailing_newline = True
            result = obj.store()
            assert "Stored" in result
            with open(tmp, "r") as f:
                assert f.read() == "replaced\n"
        finally:
            os.unlink(tmp)

    def test_store_externally_modified(self):
        """Test store() rejects writes when file was modified externally."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("original\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj._file_mtime = os.path.getmtime(tmp)
            # Simulate external modification
            os.utime(tmp, (os.path.getatime(tmp), os.path.getmtime(tmp) + 1))
            result = obj.store()
            assert "modified externally" in result
        finally:
            os.unlink(tmp)

    def test_store_without_load(self):
        """Test store() fails without prior load()."""
        obj = TextEditor()
        result = obj.store()
        assert "no file loaded" in result

    def test_store_file_deleted(self):
        """Test store() rejects writes when file no longer exists."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("content\n")
            tmp = f.name
        try:
            obj.load(tmp)
            os.unlink(tmp)
            result = obj.store()
            assert "no longer exists" in result
        finally:
            pass

    def test_diff_no_differences(self):
        """Test diff() when internal lines match the file."""
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

    def test_diff_with_differences(self):
        """Test diff() shows unified diff when lines differ."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("original\nline two\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj._lines = ["changed\nline two\n"]
            result = obj.diff()
            assert "unchanged lines match" not in result.lower()
            # Should contain unified diff markers
            assert "-" in result or "+" in result
        finally:
            os.unlink(tmp)

    def test_diff_without_load(self):
        """Test diff() fails without prior load()."""
        obj = TextEditor()
        result = obj.diff()
        assert "no file loaded" in result

    def test_clear(self):
        """Test clear() resets all state."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("content\n")
            tmp = f.name
        try:
            obj.load(tmp)
            assert obj._lines == ["content"]
            assert obj._file_path is not None
            obj.clear()
            assert obj._lines == []
            assert obj._file_path is None
            assert obj._file_mtime is None
        finally:
            os.unlink(tmp)

    def test_read_updates_mtime(self):
        """Test read() refreshes the stored mtime."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("content\n")
            tmp = f.name
        try:
            obj.load(tmp)
            mtime_after_load = obj._file_mtime
            # Wait a moment then call read
            import time
            time.sleep(0.1)
            obj.read()
            assert obj._file_mtime == mtime_after_load
        finally:
            os.unlink(tmp)


class TestTextEditorMultiStepWorkflows:
    """Manual tool-call workflows simulating what an agent would do."""

    def test_load_edit_store_roundtrip(self):
        """Load a file, edit a line, store, verify."""
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

    def test_load_write_store_roundtrip(self):
        """Load a file, write entirely new content, store, verify."""
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

    def test_load_edit_multiple_replace_all_store(self):
        """Load, replace all occurrences, store, verify."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("X\nY\nX\nZ\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj.edit("X", "W", replace_all=True)
            obj.store()
            with open(tmp, "r") as f:
                assert f.read() == "W\nY\nW\nZ\n"
        finally:
            os.unlink(tmp)

    def test_load_read_diff_no_change(self):
        """Load, read, diff — no changes means no diff."""
        obj = TextEditor()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("unchanged\nlines\n")
            tmp = f.name
        try:
            obj.load(tmp)
            obj.read()
            result = obj.diff()
            assert "No differences" in result
        finally:
            os.unlink(tmp)
