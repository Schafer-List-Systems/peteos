#!/usr/bin/env python3
"""
Interactive Shell Example

This example demonstrates how to use the InteractiveShellChannel to interact
with an Agent via a Runner.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/shell_example.py

Configure the ChatBotManager by adding your backend(s):
    await ChatBotManager.add_backend("name", "http://your-backend:port")
"""

import asyncio
import uuid
from pathlib import Path

from peteos.agent import Agent
from peteos.channels import InteractiveShellChannel
from peteos.chatbot.manager import ChatBotManager
from peteos.engine.runner import Runner
from peteos.logger import setup_logging
from peteos.role import Role
from peteos.toolmanager import ToolManager


async def setup_chatbot_manager(config_file: str = "config/chatbot_config.json"):
    """Setup ChatBotManager from configuration file."""
    ChatBotManager.reset()
    try:
        await ChatBotManager.load_from_file(config_file)
        print(f"Loaded backend configuration from {config_file}")
    except FileNotFoundError:
        print("Note: Config file not found. Starting without backends.")


def setup_role():
    """Setup the role for the agent."""
    try:
        role_dir = Path("roles")
        if role_dir.is_dir():
            for entry in sorted(role_dir.iterdir()):
                if entry.is_dir():
                    role = Role.load_from_path(str(entry))
                    print(f"Loaded role: {role.name}")
                    return role
    except FileNotFoundError:
        pass
    print("Note: No roles directory found. Creating default 'test' role.")
    return Role(name="test", description="Default test role", model=".*")


def setup_tool_manager():
    """Setup ToolManager with example tools."""
    tool_manager = ToolManager()

    def read(filename: str) -> str:
        """Read a file and return its contents as a string."""
        try:
            with open(filename, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def eval_python(python_string: str, namespace_name: str = "") -> str:
        """Execute Python code and return stdout and return_value."""
        import io
        import sys

        namespaces: dict[str, dict] = {}
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

    return tool_manager


async def main():
    """Main entry point."""
    setup_logging(level="INFO", debug=False)

    print("=" * 60)
    print("  Peteos Interactive Shell Example")
    print("=" * 60)
    print()

    # Setup components
    role = setup_role()
    await setup_chatbot_manager()
    tool_manager = setup_tool_manager()

    # Create the Agent
    agent = Agent(role, tool_manager)

    # Create the first session and runner
    session = await agent.create_session()
    runner = Runner(agent, session.uuid)
    await runner.start()

    # Create the shell channel
    shell = InteractiveShellChannel("shell", runner)

    print("-" * 60)
    print()
    print("Available commands:")
    print("  /approve  - Approve the first pending tool call")
    print("  /deny     - Deny the first pending tool call")
    print("  /pending  - List pending tool calls")
    print("  /image <filepath> [text] - Attach an image or file")
    print("  /quit     - Exit the shell")
    print()
    print("Type any text (without /) to send a message to the runner.")
    print()
    print("-" * 60)
    print()

    # Start the shell (creates input loop task) and then run (blocks until /quit)
    await shell.start()
    await shell.run()

    print()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
