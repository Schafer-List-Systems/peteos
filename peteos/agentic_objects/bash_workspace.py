"""BashWorkspace - secure bash execution agentic object."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

if TYPE_CHECKING:
    from peteos.engine import Runner

# Safe POSIX utilities available in the sandbox.
# No network tools (curl, wget, nc), no shells (bash -c via env vars), no
# process control (kill, top), no system introspection (ps, whoami).
_SAFE_COMMANDS: set[str] = {
    "cat", "echo", "grep", "egrep", "sed", "awk", "sort", "uniq", "wc",
    "head", "tail", "cut", "tr", "mkdir", "cp", "mv", "rm", "ln", "chmod",
    "touch", "find", "basename", "dirname", "test", "date", "sleep",
    "tee", "xargs", "shuf", "paste", "join", "diff", "comm", "uniq",
    "base64", "md5sum", "sha256sum", "stat", "du", "file", "ls",
    # Bash builtins — executed via /bin/sh -c, so not listed here directly.
}

# Strictly minimal POSIX environment: no PATH leaks, no secrets, no user config.
_MINIMAL_ENV: dict[str, str] = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp/bash_workspace_home",
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
    "TMPDIR": "/tmp",
}


class BashWorkspace(AgenticObjectBase):
    """You are a secure bash workspace for executing shell commands.

    You can run bash commands on files placed in your workspace. You have
    NO access to the rest of the filesystem beyond your workspace directory.
    The environment is minimal: no network tools, no secrets, no user data.

    Your workspace is created fresh for each invocation and destroyed
    afterward. Files must be provided to you — you cannot access anything
    outside your workspace directory.

    Available commands: cat, echo, grep, sed, awk, sort, uniq, wc, head,
    tail, cut, tr, mkdir, cp, mv, rm, ln, chmod, touch, find, basename,
    dirname, test, date, sleep, tee, xargs, shuf, paste, join, diff, comm,
    base64, md5sum, sha256sum, stat, du, file, ls.

    No network tools (curl, wget, nc) or shell escapes are available.
    """

    def __init__(self) -> None:
        super().__init__()
        self._workspace_dir: Path | None = None

    @property
    def workspace_dir(self) -> Path | None:
        """The current workspace directory, or None if not active."""
        return self._workspace_dir

    def _setup_workspace(self) -> Path:
        """Create or return the workspace directory.

        Uses an existing workspace if available, otherwise creates a new
        temporary directory.
        """
        if self._workspace_dir is None or not self._workspace_dir.is_dir():
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="bash_workspace_"))
            # Create HOME dir so tools like `ssh` (if it slipped through)
            # don't error on missing ~/.ssh — this is a defense-in-depth.
            (self._workspace_dir / "home").mkdir(exist_ok=True)
            os.environ["HOME"] = str(self._workspace_dir / "home")
        return self._workspace_dir

    def _teardown_workspace(self) -> None:
        """Remove the workspace directory."""
        if self._workspace_dir and self._workspace_dir.is_dir():
            import shutil
            shutil.rmtree(self._workspace_dir, ignore_errors=True)
        self._workspace_dir = None

    @property
    def _env(self) -> dict[str, str]:
        """Build the sandbox environment with the workspace HOME."""
        env = {**_MINIMAL_ENV}
        if self._workspace_dir:
            env["HOME"] = str(self._workspace_dir / "home")
        # Inherit only a few safe env vars from the parent process.
        for key in ("TERM", "LANG", "LC_ALL"):
            val = os.environ.get(key)
            if val:
                env[key] = val
        return env

    def _is_safe_command(self, command: str) -> bool:
        """Check if the command (first word) is in the allowlist."""
        first = command.strip().split()[0] if command.strip() else ""
        # Handle built-in /bin/sh -c wrapper
        if first == "/bin/sh" or first == "sh":
            return True
        # Extract basename for /usr/bin/foo paths
        base = os.path.basename(first)
        return base in _SAFE_COMMANDS

    @tool
    def bash_exec(self, command: str) -> str:
        """Execute a bash command in the sandboxed workspace.

        Args:
            command: The bash command to execute.

        Returns:
            A string containing stdout, stderr, and exit code information.
        """
        ws = self._setup_workspace()

        if not command.strip():
            return "Error: Empty command."

        # Block path traversal patterns in the raw command text before
        # we even spawn a shell. This catches ../ escape attempts early.
        if ".." in command:
            return "Error: Path traversal is not allowed."

        # Block command substitution and backticks that could bypass
        # the allowlist via subshell execution.
        _BLOCKED_CHARS = ("`", "$(")
        for bad in _BLOCKED_CHARS:
            if bad in command:
                return f"Error: The character or sequence '{bad}' is not allowed."

        # Quick allowlist check on the first token to catch obviously
        # dangerous commands before we even spawn a shell.
        if not self._is_safe_command(command):
            base = os.path.basename(command.strip().split()[0])
            return f"Error: Command '{base}' is not allowed in the sandbox."

        try:
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=ws,
                env=self._env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = ""
            if result.stdout:
                output += f"stdout:\n{result.stdout}"
            if result.stderr:
                output += f"stderr:\n{result.stderr}"
            if not output:
                output = "(no output)"
            output += f"\nexit_code: {result.returncode}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool
    def put_file(self, name: str, content: str) -> str:
        """Place a file into the workspace.

        Args:
            name: The filename to create in the workspace.
            content: The file contents as a string.

        Returns:
            Confirmation message.
        """
        ws = self._setup_workspace()
        filepath = (ws / name).resolve()
        # Guard: prevent directory traversal outside workspace.
        # resolve() normalizes .. so this is a proper containment check.
        try:
            filepath.relative_to(ws.resolve())
        except ValueError:
            return f"Error: File name '{name}' is not allowed."
        try:
            filepath.write_text(content)
            return f"OK: File '{name}' placed in workspace."
        except Exception as e:
            return f"Error: Failed to write '{name}': {e}"

    @tool
    def get_file(self, name: str) -> str:
        """Read the content of a file from the workspace.

        Args:
            name: The filename to read from the workspace.

        Returns:
            The file contents as a string.
        """
        ws = self._setup_workspace()
        filepath = (ws / name).resolve()
        # resolve() normalizes .. so this is a proper containment check.
        try:
            filepath.relative_to(ws.resolve())
        except ValueError:
            return f"Error: File name '{name}' is not allowed."
        if not filepath.is_file():
            return f"Error: File '{name}' not found in workspace."
        try:
            return filepath.read_text()
        except Exception as e:
            return f"Error: Failed to read '{name}': {e}"
