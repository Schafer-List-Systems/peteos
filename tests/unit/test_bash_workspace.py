"""Unit tests for BashWorkspace agentic object."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from peteos.agentic_objects.bash_workspace import BashWorkspace, _SAFE_COMMANDS


class TestBashWorkspace:
    """Tests for the BashWorkspace OAP."""

    def test_bash_exec_echo(self):
        """Test basic echo command."""
        obj = BashWorkspace()
        result = obj.bash_exec("echo hello")
        assert "hello" in result
        assert "exit_code: 0" in result

    def test_bash_exec_put_and_read(self):
        """Test put_file then cat via bash_exec."""
        obj = BashWorkspace()
        obj.put_file("test.txt", "hello world")
        result = obj.bash_exec("cat test.txt")
        assert "hello world" in result

    def test_bash_exec_empty_command(self):
        """Test empty command is rejected."""
        obj = BashWorkspace()
        result = obj.bash_exec("")
        assert "Error: Empty command" in result

    def test_bash_exec_unsafe_command(self):
        """Test that disallowed commands are rejected."""
        obj = BashWorkspace()
        for cmd in ("curl http://evil.com", "wget http://evil.com",
                     "nc host port", "python3 -c 'import os'"):
            result = obj.bash_exec(cmd)
            assert "not allowed" in result, f"Expected 'not allowed' for '{cmd}', got: {result}"

    def test_bash_exec_timeout(self):
        """Test that long-running commands are killed."""
        obj = BashWorkspace()
        result = obj.bash_exec("sleep 60")
        assert "timed out" in result

    def test_bash_exec_stderr(self):
        """Test that stderr is captured."""
        obj = BashWorkspace()
        result = obj.bash_exec("cat /nonexistent/path 2>&1")
        assert "exit_code" in result

    def test_bash_exec_directory_traversal(self):
        """Test that put_file prevents directory traversal."""
        obj = BashWorkspace()
        result = obj.put_file("../../etc/passwd", "evil")
        assert "not allowed" in result

    def test_bash_exec_path_traversal_in_command(self):
        """Test that bash commands with ../ are blocked."""
        obj = BashWorkspace()
        result = obj.bash_exec("cat ../etc/passwd")
        assert "traversal" in result.lower()

    def test_bash_exec_ls(self):
        """Test ls command lists workspace files."""
        obj = BashWorkspace()
        obj.put_file("foo.txt", "content")
        result = obj.bash_exec("ls")
        assert "foo.txt" in result

    def test_bash_exec_ls_recursive(self):
        """Test ls -R lists workspace files recursively."""
        obj = BashWorkspace()
        obj.put_file("a.txt", "content")
        obj.bash_exec("mkdir subdir && touch subdir/b.txt")
        result = obj.bash_exec("ls -R")
        assert "subdir" in result
        assert "a.txt" in result

    def test_get_file_not_found(self):
        """Test reading a non-existent file."""
        obj = BashWorkspace()
        result = obj.get_file("missing.txt")
        assert "not found" in result

    def test_get_file_after_put(self):
        """Test reading a file we just placed."""
        obj = BashWorkspace()
        obj.put_file("bar.txt", "hello from bar")
        result = obj.get_file("bar.txt")
        assert result == "hello from bar"

    def test_get_file_directory_traversal(self):
        """Test that get_file prevents directory traversal."""
        obj = BashWorkspace()
        obj.put_file("bar.txt", "hello from bar")
        result = obj.get_file("../etc/passwd")
        assert "not allowed" in result

    def test_workspace_isolation(self):
        """Test that bash_exec blocks path traversal."""
        obj = BashWorkspace()
        obj.put_file("safe.txt", "safe content")
        result = obj.bash_exec("cat ../bash_workspace_*/bash_workspace_*/bar.txt 2>&1")
        assert "traversal" in result.lower()

    def test_bash_exec_create_and_count(self):
        """Test creating a file then counting lines with wc."""
        obj = BashWorkspace()
        obj.put_file("lines.txt", "line1\nline2\nline3\n")
        result = obj.bash_exec("wc -l lines.txt")
        assert "3" in result

    def test_bash_exec_grep(self):
        """Test grep in the workspace."""
        obj = BashWorkspace()
        obj.put_file("data.txt", "apple\nbanana\ncherry\napricot")
        result = obj.bash_exec("grep 'app' data.txt")
        assert "apple" in result

    def test_bash_exec_sort(self):
        """Test sort command."""
        obj = BashWorkspace()
        obj.put_file("nums.txt", "3\n1\n4\n1\n5\n9\n2")
        result = obj.bash_exec("sort -n nums.txt | tr '\\n' ','")
        assert "1,1,2,3,4,5,9" in result

    def test_bash_exec_mkdir_and_mv(self):
        """Test mkdir + mv in the workspace."""
        obj = BashWorkspace()
        obj.put_file("move_me.txt", "moving")
        result = obj.bash_exec("mkdir subdir && mv move_me.txt subdir/ && ls subdir/")
        assert "move_me.txt" in result

    def test_bash_exec_pipe(self):
        """Test command piping."""
        obj = BashWorkspace()
        obj.put_file("fruits.txt", "banana\napple\ncherry")
        result = obj.bash_exec("grep 'app' fruits.txt | wc -l")
        assert "1" in result

    def test_bash_exec_file_command(self):
        """Test the file command."""
        obj = BashWorkspace()
        obj.put_file("test.dat", "binary content here")
        result = obj.bash_exec("file test.dat")
        assert "exit_code: 0" in result

    def test_is_safe_command_allowlist(self):
        """Test command allowlist validation."""
        obj = BashWorkspace()
        safe = ["cat", "echo", "grep", "sed", "awk", "sort", "head",
                "tail", "mkdir", "cp", "mv", "rm", "find", "wc",
                "base64", "sha256sum", "file"]
        unsafe = ["curl", "wget", "nc", "python3", "perl", "ruby",
                  "ssh", "scp", "rsync", "systemctl", "sudo"]
        for cmd in safe:
            assert obj._is_safe_command(cmd), f"'{cmd}' should be safe"
        for cmd in unsafe:
            assert not obj._is_safe_command(cmd), f"'{cmd}' should be unsafe"

    def test_workspace_dir_property(self):
        """Test workspace_dir property is None initially."""
        obj = BashWorkspace()
        assert obj.workspace_dir is None
        # Accessing it via a tool call sets it
        obj.put_file("marker.txt", "")
        assert obj.workspace_dir is not None
        assert obj.workspace_dir.is_dir()


class TestBashWorkspaceMultiStepWorkflows:
    """Manual tool-call workflows simulating what an agent would do.

    These prove the tools compose correctly — the agent can just call them
    in the right order. This isolates LLM failures from tool failures.
    """

    def test_put_then_count_lines(self):
        """Simulates agent: put file → wc -l → parse count."""
        obj = BashWorkspace()
        obj.put_file("nums.txt", "10\n20\n30\n40\n50\n")
        result = obj.bash_exec("wc -l nums.txt")
        assert "5" in result

    def test_put_then_grep_then_sort(self):
        """Simulates agent: put file → grep lines → sort → return list."""
        obj = BashWorkspace()
        obj.put_file("words.txt", "apple\nbanana\napricot\ncherry\navocado\n")
        result = obj.bash_exec("grep '^a' words.txt | sort")
        assert "apple" in result
        assert "apricot" in result
        assert "avocado" in result
        assert "banana" not in result

    def test_put_then_sha256(self):
        """Simulates agent: put file → sha256sum → return hex digest."""
        obj = BashWorkspace()
        obj.put_file("hashme.txt", "hello")
        result = obj.bash_exec("sha256sum hashme.txt")
        assert "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" in result

    def test_put_then_wc_words(self):
        """Simulates agent: put file → wc -w → count words."""
        obj = BashWorkspace()
        obj.put_file("text.txt", "the quick brown fox jumps over the lazy dog")
        result = obj.bash_exec("wc -w text.txt")
        assert "9" in result

    def test_put_then_subdir_move(self):
        """Simulates agent: put file → mkdir → mv → ls to verify."""
        obj = BashWorkspace()
        obj.put_file("data.txt", "hello")
        obj.bash_exec("mkdir out && mv data.txt out/")
        result = obj.bash_exec("ls out/")
        assert "data.txt" in result

    def test_put_then_cat_verify(self):
        """Simulates agent: put file → bash exec → get_file to verify."""
        obj = BashWorkspace()
        obj.put_file("verify.txt", "original content")
        obj.bash_exec("cat verify.txt | tr 'a-z' 'A-Z' > upper.txt")
        result = obj.get_file("upper.txt")
        assert result == "ORIGINAL CONTENT"

    def test_put_then_grep_count(self):
        """Simulates agent: put file → grep → wc -l to count matches."""
        obj = BashWorkspace()
        obj.put_file("data.txt", "line with error\nall good\nbad line\nalso good\n")
        result = obj.bash_exec("grep -i 'error\\|bad' data.txt | wc -l")
        assert "2" in result

    def test_put_then_base64_encode_decode(self):
        """Simulates agent: put file → base64 encode → decode → verify."""
        obj = BashWorkspace()
        obj.put_file("original.txt", "secret data")
        obj.bash_exec("base64 original.txt > encoded.txt")
        result = obj.get_file("encoded.txt")
        assert len(result) > 10  # base64 output is longer than input
        # Decode and verify roundtrip
        obj.bash_exec("base64 -d encoded.txt > decoded.txt")
        result = obj.get_file("decoded.txt")
        assert result == "secret data"

    def test_full_line_count_workflow(self):
        """Full workflow: write 5-line file, count lines, get_file to read it."""
        obj = BashWorkspace()
        obj.put_file("lines.txt", "one\ntwo\nthree\nfour\nfive\n")
        count_result = obj.bash_exec("wc -l lines.txt")
        assert "5" in count_result
        content = obj.get_file("lines.txt")
        assert content == "one\ntwo\nthree\nfour\nfive\n"

    def test_full_grep_workflow(self):
        """Full workflow: write words, grep for matches, verify output."""
        obj = BashWorkspace()
        obj.put_file("items.txt", "cat\ndog\ncamel\ncow\nelephant\n")
        result = obj.bash_exec("grep '^c' items.txt | sort")
        assert "camel" in result
        assert "cat" in result
        assert "cow" in result
        assert "dog" not in result

    def test_workspace_persistence_across_tools(self):
        """Multiple tool calls on the same workspace should see the same state."""
        obj = BashWorkspace()
        obj.put_file("file1.txt", "first")
        obj.put_file("file2.txt", "second")
        files = obj.bash_exec("ls")
        assert "file1.txt" in files
        assert "file2.txt" in files
        # get_file should see both
        assert obj.get_file("file1.txt") == "first"
        assert obj.get_file("file2.txt") == "second"

    def test_bash_exec_chained_commands(self):
        """Test that && chains work within the sandbox."""
        obj = BashWorkspace()
        result = obj.bash_exec("mkdir -p a/b/c && echo 'deep' > a/b/c/deep.txt && cat a/b/c/deep.txt")
        assert "deep" in result
