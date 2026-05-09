#!/usr/bin/env python3
"""
Interactive Shell Example

This example demonstrates how to use the InteractiveShellChannel to interact
with an Agent using the queue-based architecture.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/shell_example.py

Configure the ChatBotManager in setup_components() by adding your backend(s):
    await chatbot_manager.add_backend("name", "http://your-backend:port")
"""

from peteos.agent import Agent
from peteos.channels import InteractiveShellChannel
from peteos.chatbot.manager import ChatBotManager
from peteos.logger import setup_logging
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


async def setup_chatbot_manager(config_file: str = "config/chatbot_config.json"):
    """Setup ChatBotManager from configuration file.

    Args:
        config_file: Path to JSON configuration file with backend definitions.

    Returns:
        Configured ChatBotManager with all backends loaded.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        RuntimeError: If backend connection fails.
    """
    chatbot_manager = ChatBotManager()

    try:
        await chatbot_manager.load_from_file(config_file)
        print(f"Loaded backend configuration from {config_file}")
    except FileNotFoundError:
        print(f"Note: Config file {config_file} not found. Starting without backends.")
    except RuntimeError as e:
        print(f"Error: Backend connection failed: {e}")
        raise

    return chatbot_manager


def setup_role_manager():
    """Setup RoleManager with available roles.

    Returns:
        Configured RoleManager with roles loaded from roles/ directory.
    """
    role_manager = RoleManager()

    try:
        loaded_roles = role_manager.load_from_dir("roles")
        print(f"Loaded roles: {', '.join(loaded_roles)}")
    except FileNotFoundError:
        print("Note: No roles directory found. Creating default 'test' role.")
        role_manager.register_role(
            Role(name="test", description="Default test role", model=".*")
        )

    return role_manager


def setup_tool_manager():
    """Setup ToolManager with example tools.

    Returns:
        Configured ToolManager with read and eval_python tools.
    """
    tool_manager = ToolManager()
    namespaces: dict[str, dict] = {}

    def read(filename: str) -> str:
        """Read a file and return its contents as a string.

        Args:
            filename: The path to the file to read.
        """
        try:
            with open(filename, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def eval_python(python_string: str, namespace_name: str = "") -> str:
        """Execute Python code and return stdout and return_value.

        The return value is captured by setting _result in the code.
        Use the same namespace_name across calls to maintain state (variables defined in one call are available in subsequent calls).
        Omit namespace_name or pass '' for a fresh anonymous namespace destroyed after each call.
        Pass 'globals' to execute in the module's global namespace (sharing module-level imports and definitions).
        Pass a named namespace_name for persistent state.

        Args:
            python_string: A string containing valid Python code to execute.
            namespace_name: The namespace name for state persistence. Empty string for ephemeral (default).
        """
        import io
        import sys

        if namespace_name == "globals":
            ns: dict = globals()
        elif namespace_name == "":
            ns = {}
        else:
            ns = namespaces.get(namespace_name)
            if ns is None:
                namespaces[namespace_name] = {}
                ns = namespaces[namespace_name]

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        return_value = None
        try:
            sys.stdout = stdout_capture
            code = compile(python_string, "<eval>", "exec")
            exec(code, ns)
            return_value = ns.get("_result")
        except Exception as e:
            return_value = f"Error: {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout

        stdout = stdout_capture.getvalue()
        return f"stdout: {stdout!r}\nreturn_value: {return_value!r}"

    tool_manager.register_tool(func=read)
    tool_manager.register_tool(func=eval_python)

    return tool_manager


async def main():
    """Main entry point."""
    # Configure logging
    setup_logging(level="INFO", debug=False)

    print("=" * 60)
    print("  Peteos Interactive Shell Example")
    print("=" * 60)
    print()

    # Setup components
    role_manager = setup_role_manager()
    chatbot_manager = await setup_chatbot_manager()
    tool_manager = setup_tool_manager()

    # Create the Agent
    agent = Agent(role_manager, chatbot_manager, tool_manager)

    # Create the shell channel
    shell = InteractiveShellChannel("shell", agent)

    print("-" * 60)
    print()
    print("Available commands:")
    print("  /new <role>  - Create a new session")
    print("  /list        - List all sessions")
    print("  /select <uuid> - Select a session as active")
    print("  /messages    - Show recent messages")
    print("  /quit        - Exit the shell")
    print()
    print("Type any text (without /) to send a message to the active session.")
    print()
    print("-" * 60)
    print()

    # Start the shell (creates input loop task) and then run (blocks until /quit)
    await shell.start()
    await shell.run()

    print()
    print("Goodbye!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
