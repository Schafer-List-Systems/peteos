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


class TestBashWorkspaceIntegration:
    """End-to-end tests using invoke_agent with a mock backend.

    These tests require a live LLM backend (same as PdfTranscriber tests).
    """

    @pytest.mark.oap
    async def test_agent_can_read_and_transform_file(self):
        """Test the agent can use put_file + bash_exec + get_file together."""
        obj = BashWorkspace()
        result = await obj.invoke_agent(
            prompt=(
                "Place a file called 'numbers.txt' with content "
                "'10\\n20\\n30\\n40\\n50'. Then count how many lines "
                "it has using bash and put the count into 'count.txt'. "
                "Return the content of count.txt using produce_output."
            ),
            output_schema=str,
            timeout=60,
        )
        assert isinstance(result, (str, Error))
        if isinstance(result, str):
            assert "5" in result
