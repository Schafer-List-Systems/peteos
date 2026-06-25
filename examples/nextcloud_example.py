#!/usr/bin/env python3
"""
Nextcloud Talk Bot Example

This example demonstrates how to use the NextcloudTalkChannel to connect
an Agent to a Nextcloud Talk room as a bot.

Prerequisites:
  1. Create a bot in your Nextcloud instance:
     ./occ talk:bot:install <bot-name> <webhook-url> <secret>
  2. Configure the chatbot backend (your LLM) in examples/config/chatbot_config.json
  3. Configure the Nextcloud Talk bot in examples/config/nextcloud_config.json

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      examples/nextcloud_example.py

    # Override config paths:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      examples/nextcloud_example.py \
      --nextcloud-config /path/to/nextcloud_config.json

When you run this, the channel starts an HTTP server. Visit its URL in a
browser or use curl to verify it's listening, then point your Nextcloud
bot's webhook URL at <server-url>/webhook.
"""

import argparse
import asyncio
import json
import sys

from peteos.agent import Agent
from peteos.channels import NextcloudTalkChannel
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
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
        print(f"Note: Config file {config_file} not found. Starting without backends.")
    except RuntimeError as e:
        print(f"Error: Backend connection failed: {e}")
        raise


def setup_role() -> Role:
    """Setup the role for the agent.

    Returns:
        Role loaded from roles/ directory, or default 'test' role.
    """
    from pathlib import Path

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


def load_nextcloud_config(config_file: str = "examples/config/nextcloud_config.json"):
    """Load Nextcloud Talk bot configuration from a JSON file.

    Delegates to NextcloudTalkChannel.load_config() with error message formatting.
    """
    try:
        return NextcloudTalkChannel.load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        print("Create it from the template:")
        print(f"  cp examples/config/nextcloud_config.json.example {config_file}")
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Peteos Nextcloud Talk Bot")
    parser.add_argument(
        "--chatbot-config",
        default="config/chatbot_config.json",
        help="Path to chatbot backend config (default: config/chatbot_config.json)",
    )
    parser.add_argument(
        "--nextcloud-config",
        default="examples/config/nextcloud_config.json",
        help="Path to Nextcloud Talk bot config (default: examples/config/nextcloud_config.json)",
    )
    args = parser.parse_args()

    # Configure logging
    setup_logging(level="INFO", debug=True)

    print("=" * 60)
    print("  Peteos Nextcloud Talk Bot Example")
    print("=" * 60)
    print()

    # Setup components
    role = setup_role()
    await setup_chatbot_manager(args.chatbot_config)
    tool_manager = setup_tool_manager()

    # Create the Agent
    agent = Agent(role, tool_manager)

    # Load Nextcloud configuration
    try:
        config = load_nextcloud_config(args.nextcloud_config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Aborting: {e}")
        return

    # Create a session and runner for the default room
    session = await agent.create_session()
    runner = Runner(agent, session.uuid)
    await runner.start()
    print(f"Created session: {session.uuid}")
    print()

    # Create the Nextcloud Talk channel connected to the runner
    nextcloud = NextcloudTalkChannel(name="nextcloud", runner=runner, config=config)

    # Start the webhook receiver
    webhook_url = await nextcloud.start()
    print(f"Webhook receiver started at: {webhook_url}")
    print()
    print("Next, register this URL with your Nextcloud bot:")
    print()
    print(f"  ./occ talk:bot:install <name> {webhook_url}/webhook <secret>")
    print()
    print("Or manually set the webhook URL in the bot configuration.")
    print()
    print("The bot will automatically create sessions when added to rooms.")
    print("Responses are sent back to the originating room.")
    print()

    # Handle pre-joined rooms from config
    auto_join_rooms = config.get("auto_join_rooms", [])
    for room_token in auto_join_rooms:
        await nextcloud.register_room(session.uuid, room_token)
        print(f"Registered room {room_token} with session {session.uuid}")

    # When the bot is added to a new room, the channel calls this callback
    # to let the app create a new runner+channel for that room
    async def on_room_joined(room_token: str):
        """Handle new room join: create new session, runner, channel, then register."""
        new_session = await agent.create_session()
        new_runner = Runner(agent, new_session.uuid)
        await new_runner.start()
        new_channel = NextcloudTalkChannel(
            name=f"nextcloud-{room_token}",
            runner=new_runner,
            config=config,
        )
        new_channel.on_room_joined = on_room_joined
        await new_channel.register_room(new_session.uuid, room_token)
        await new_channel.start()
        print(f"Registered new room {room_token} with session {new_session.uuid}")

    nextcloud.on_room_joined = on_room_joined

    # Wait for webhook events (blocks until stopped)
    try:
        print("Listening for webhook events... Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nSending goodbye message...")
        if nextcloud._rooms:
            for token, session_uuid in nextcloud._rooms.items():
                bye = Message.create(
                    role="assistant",
                    content_parts=[ContentPart.create_text("I am going offline.")],
                )
                await nextcloud.send(bye, session_uuid=session_uuid)
                print(f"  Sent 'I am going offline.' to room {token}")
        await nextcloud.stop()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
