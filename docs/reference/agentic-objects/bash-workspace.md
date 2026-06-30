# BashWorkspace

A secure bash workspace agentic object that provides a sandboxed environment for executing shell commands.

## Description

`BashWorkspace` is an agentic object addressed to the agent rather than the user. It is not required for normal usage — it exists solely to give the agent the ability to execute commands in a restricted, sandboxed bash environment.

The workspace is created fresh for each invocation and destroyed afterward. It provides:
- A temporary directory as the workspace root.
- A minimal POSIX environment (no PATH leaks, no secrets, no user data).
- An allowlist of safe POSIX utilities only.
- Path traversal and command injection protections.

## Tools

### `bash_exec(command: str) -> str`

Execute a bash command in the sandboxed workspace.

**Args:**
- `command` — The bash command to execute.

**Returns:** A string containing exit code, stdout, and stderr information.

**Security measures:**
- Path traversal (`..`) is blocked.
- Command substitution and backticks (`\`` and `$(`) are blocked.
- Only commands in the safe allowlist are permitted.
- The sandbox has no network tools, no shell escapes, and no process control utilities.

**Safe commands:** `cat`, `echo`, `grep`, `egrep`, `sed`, `awk`, `sort`, `uniq`, `wc`, `head`, `tail`, `cut`, `tr`, `mkdir`, `cp`, `mv`, `rm`, `ln`, `chmod`, `touch`, `find`, `basename`, `dirname`, `test`, `date`, `sleep`, `tee`, `xargs`, `shuf`, `paste`, `join`, `diff`, `comm`, `base64`, `md5sum`, `sha256sum`, `stat`, `du`, `file`, `ls`.

Commands timeout after 30 seconds.

### `put_file(name: str, content: str) -> str`

Place a file into the workspace.

**Args:**
- `name` — The filename to create in the workspace.
- `content` — The file contents as a string.

**Returns:** Confirmation message.

Files are prevented from escaping the workspace directory via path resolution checks.

### `get_file(name: str) -> str`

Read the content of a file from the workspace.

**Args:**
- `name` — The filename to read from the workspace.

**Returns:** The file contents as a string.

Files outside the workspace directory are rejected.

## Usage

To use the BashWorkspace, derive from it as a parent class of your agentic object:

```python
class MyAgent(BashWorkspace, AgenticObjectBase):
    """You are an agent with a secure bash workspace."""
```

The agent will have access to the `bash_exec`, `put_file`, and `get_file` tools through its system prompt.
