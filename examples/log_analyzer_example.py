#!/usr/bin/env python3
"""
Log Analyzer Example

This example demonstrates how to use a read-only stdout channel to monitor
system logs (e.g., journalctl) and connect them to an agent via Nextcloud Talk.

A session is created upfront, and two channels attach to it:
  1. Nextcloud Talk channel - the user-facing channel for interaction
  2. Stdout channel - reads journalctl output, forwards matching lines to the session

Prerequisites:
  1. Create a bot in your Nextcloud instance:
     ./occ talk:bot:install <bot-name> <webhook-url> <secret>
  2. Configure the chatbot backend (your LLM) in examples/config/chatbot_config.json
  3. Configure the Nextcloud Talk bot in examples/config/nextcloud_config.json
  4. Configure the stdout monitor in examples/config/log_analyzer_config.json

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      examples/log_analyzer_example.py

    # Override config paths:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      examples/log_analyzer_example.py \
      --nextcloud-config /path/to/nextcloud_config.json \
      --stdout-config /path/to/log_analyzer_config.json

When you run this, the channel starts an HTTP server. Visit its URL in a
browser or use curl to verify it's listening, then point your Nextcloud
bot's webhook URL at <server-url>/webhook.
"""

import argparse
import asyncio
import re
import sys
import uuid

from peteos.agent import Agent
from peteos.channels import NextcloudTalkChannel, ReadStdoutChannel
from peteos.chatbot.manager import ChatBotManager
from peteos.chatbot import Message, ContentPart
from peteos.logger import setup_logging
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager


async def setup_chatbot_manager(config_file: str = "config/chatbot_config.json"):
    """Setup ChatBotManager from configuration file."""
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
    """Setup RoleManager with available roles."""
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
    """Setup ToolManager with example tools."""
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
    """Load Nextcloud Talk bot configuration from a JSON file."""
    try:
        return NextcloudTalkChannel.load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        print("Create it from the template:")
        print(f"  cp examples/config/nextcloud_config.json.example {config_file}")
        raise


def load_stdout_config(config_file: str = "examples/config/log_analyzer_config.json"):
    """Load stdout channel configuration from a JSON file."""
    try:
        return ReadStdoutChannel.load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        print("Create it from the template:")
        print(f"  cp examples/config/log_analyzer_config.json.example {config_file}")
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Peteos Log Analyzer Example")
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
    parser.add_argument(
        "--stdout-config",
        default="examples/config/log_analyzer_config.json",
        help="Path to stdout channel config (default: examples/config/log_analyzer_config.json)",
    )
    args = parser.parse_args()

    # Configure logging
    setup_logging(level="INFO", debug=True)

    print("=" * 60)
    print("  Peteos Log Analyzer Example")
    print("=" * 60)
    print()

    # Setup components
    role_manager = setup_role_manager()
    chatbot_manager = await setup_chatbot_manager(args.chatbot_config)
    tool_manager = setup_tool_manager()

    # Create the Agent
    agent = Agent(role_manager, chatbot_manager, tool_manager)

    # Load configurations
    try:
        nextcloud_config = load_nextcloud_config(args.nextcloud_config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Aborting: {e}")
        return

    try:
        stdout_config = load_stdout_config(args.stdout_config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Aborting: {e}")
        return

    # Create a shared session that both channels attach to
    role_name = nextcloud_config.get("default_role", "test")
    session = await agent.create_session(role_name)
    print(f"Created session: {session.uuid}")
    print()

    # Create the Nextcloud Talk channel (user-facing, sends and receives)
    nextcloud = NextcloudTalkChannel(name="nextcloud", agent=agent, config=nextcloud_config)

    # Register rooms with the session (app owns session lifecycle)
    auto_join_rooms = nextcloud_config.get("auto_join_rooms", [])
    for room_token in auto_join_rooms:
        await nextcloud.register_room(session.uuid, room_token)
        print(f"Registered room {room_token} with session {session.uuid}")

    # Callback for dynamic room joins
    async def on_room_joined(room_token: str):
        new_session = await agent.create_session(role_name)
        await nextcloud.register_room(new_session.uuid, room_token)
        print(f"Registered new room {room_token} with session {new_session.uuid}")

    nextcloud.on_room_joined = on_room_joined

    # Create the stdout monitor channel (read-only, forwards log lines to session)
    stdout_channel = ReadStdoutChannel(
        name="log-monitor",
        agent=agent,
        config=stdout_config,
    )
    stdout_channel.subscribe_to_session(session.uuid)

    # Start both channels
    await stdout_channel.start()

    webhook_url = await nextcloud.start()
    print(f"Webhook receiver started at: {webhook_url}")
    print()
    print("Next, register this URL with your Nextcloud bot:")
    print()
    print(f"  ./occ talk:bot:install <name> {webhook_url}/webhook <secret>")
    print()
    print("Or manually set the webhook URL in the bot configuration.")
    print()
    print(f"Stdout monitor started: {stdout_config['command']}")
    print(f"  Pattern: {stdout_config.get('pattern', '.*')}")
    print()
    print("The stdout channel will monitor logs and forward matching lines")
    print("to the session. Responses go back through the Nextcloud channel.")
    print()

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
                await nextcloud.send(
                    Message(role="assistant", content=[ContentPart(part_type="text", text="I am going offline.")]),
                    session_uuid=session_uuid,
                )
                print(f"  Sent 'I am going offline.' to room {token}")
        await nextcloud.stop()
        await stdout_channel.stop()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
