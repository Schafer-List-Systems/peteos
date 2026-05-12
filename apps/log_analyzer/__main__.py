#!/usr/bin/env python3
"""
Log Analyzer App

This app monitors system logs (e.g., journalctl) and categorizes them
using a Router agent. It uses two channels attached to the same session:
  1. Nextcloud Talk channel - the user-facing channel for interaction
  2. Stdout channel - reads journalctl output, forwards matching lines to the session

Prerequisites:
  1. Create a bot in your Nextcloud instance:
     ./occ talk:bot:install <bot-name> <webhook-url> <secret>
  2. Configure the chatbot backend (your LLM) in examples/config/chatbot_config.json
  3. Configure the Nextcloud Talk bot in apps/log_analyzer/config/nextcloud_config.json
  4. Configure the stdout monitor in apps/log_analyzer/config/stdout_config.json

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      python -m apps.log_analyzer

    # Override config paths:
    PYTHONPATH=/home/frygge/projects/private/peteos \
      python -m apps.log_analyzer \
      --nextcloud-config /path/to/nextcloud_config.json \
      --stdout-config /path/to/stdout_config.json
"""

import argparse
import asyncio
import sys
import uuid

from peteos.agent import Agent
from peteos.channels import NextcloudTalkChannel, ReadStdoutChannel
from peteos.chatbot import Message, ContentPart
from peteos.chatbot.manager import ChatBotManager
from peteos.logger import setup_logging
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager

from apps.log_analyzer.tools import LogState


async def setup_chatbot_manager(config_file: str = "examples/config/chatbot_config.json"):
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


def setup_role_manager() -> RoleManager:
    """Setup RoleManager with roles from the app's roles directory."""
    role_manager = RoleManager()

    try:
        loaded_roles = role_manager.load_from_dir("apps/log_analyzer/roles")
        print(f"Loaded roles: {', '.join(loaded_roles)}")
    except FileNotFoundError:
        print("Note: No roles directory found. Creating default 'router' role.")
        role_manager.register_role(
            Role(name="router", description="Log router role", model=".*")
        )

    return role_manager


def _register_exclude_tools(agent: Agent, stdout_channel: ReadStdoutChannel) -> None:
    """Register exclude pattern management tools on the agent's tool manager."""
    tm = agent._tool_manager

    def add_exclude_pattertn(pattern: str) -> str:
        """Add a regex exclude pattern to filter out log messages.

        Lines matching an exclude pattern are dropped. Use this to suppress unwanted noise.

        Args:
            pattern: Regular expression pattern to exclude.

        Returns:
            Status message.
        """
        if stdout_channel.add_exclude_pattern(pattern):
            return f"Added exclude pattern: {pattern!r}"
        return f"Pattern {pattern!r} already exists"

    def remove_exclude_pattern(index: int) -> str:
        """Remove an exclude pattern by its index in the list.

        Args:
            index: Position of the pattern to remove.

        Returns:
            Status message.
        """
        if stdout_channel.remove_exclude_pattern(index):
            return f"Removed pattern at index {index}"
        return f"Invalid index: {index}"

    def list_exclude_patterns() -> str:
        """List all current exclude patterns.

        Returns:
            Formatted list of active exclude patterns with their indices.
        """
        patterns = stdout_channel.list_exclude_patterns()
        if not patterns:
            return "No exclude patterns configured."
        lines = [f"  [{i}] {p!r}" for i, p in enumerate(patterns)]
        return "Exclude patterns:\n" + "\n".join(lines)

    tm.register_tool(func=add_exclude_pattern)
    tm.register_tool(func=remove_exclude_pattern)
    tm.register_tool(func=list_exclude_patterns)


def load_nextcloud_config(config_file: str = "apps/log_analyzer/nextcloud_config.json"):
    """Load Nextcloud Talk bot configuration from a JSON file."""
    try:
        return NextcloudTalkChannel.load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        print("Create it from the template:")
        print(f"  cp apps/log_analyzer/config/nextcloud_config.json.example {config_file}")
        raise


def load_stdout_config(config_file: str = "apps/log_analyzer/stdout_config.json"):
    """Load stdout channel configuration from a JSON file."""
    try:
        return ReadStdoutChannel.load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        print("Create it from the template:")
        print(f"  cp apps/log_analyzer/config/stdout_config.json.example {config_file}")
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Peteos Log Analyzer App")
    parser.add_argument(
        "--chatbot-config",
        default="apps/log_analyzer/config/chatbot_config.json",
        help="Path to chatbot backend config (default: apps/log_analyzer/config/chatbot_config.json)",
    )
    parser.add_argument(
        "--nextcloud-config",
        default="apps/log_analyzer/config/nextcloud_config.json",
        help="Path to Nextcloud Talk bot config (default: apps/log_analyzer/config/nextcloud_config.json)",
    )
    parser.add_argument(
        "--stdout-config",
        default="apps/log_analyzer/config/stdout_config.json",
        help="Path to stdout channel config (default: apps/log_analyzer/config/stdout_config.json)",
    )
    args = parser.parse_args()

    # Configure logging
    setup_logging(level="INFO", debug=True)

    print("=" * 60)
    print("  Peteos Log Analyzer App")
    print("=" * 60)
    print()

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

    # Setup components
    role_manager = setup_role_manager()
    chatbot_manager = await setup_chatbot_manager(args.chatbot_config)

    # Create the Agent
    agent = Agent(role_manager, chatbot_manager, ToolManager())

    # Create a shared session that both channels attach to
    role_name = nextcloud_config.get("default_role", "router")
    session = await agent.create_session(role_name)
    print(f"Created session: {session.uuid}")
    print()

    # Create the Nextcloud Talk channel (user-facing, sends and receives)
    nextcloud = NextcloudTalkChannel(name="nextcloud", agent=agent, config=nextcloud_config)

    # Register rooms with the session (app owns session lifecycle)
    auto_join_rooms = nextcloud_config.get("auto_join_rooms", [])
    for room_token in auto_join_rooms:
        nextcloud.register_room(session.uuid, room_token)
        print(f"Registered room {room_token} with session {session.uuid}")

    # Callback for dynamic room joins
    async def on_room_joined(room_token: str):
        new_session = await agent.create_session(role_name)
        nextcloud.register_room(new_session.uuid, room_token)
        print(f"Registered new room {room_token} with session {new_session.uuid}")

    nextcloud.on_room_joined = on_room_joined

    # Create the stdout monitor channel (read-only, forwards log lines to session)
    stdout_channel = ReadStdoutChannel(
        name="log-monitor",
        agent=agent,
        config=stdout_config,
    )
    stdout_channel.subscribe_to_session(session.uuid)

    # Register stdout channel exclude pattern tools with the agent's tool manager
    _register_exclude_tools(agent, stdout_channel)

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
                nextcloud.send(
                    Message(role="assistant", content=[ContentPart(part_type="text", text="I am going offline.")]),
                    session_uuid=session_uuid,
                )
                print(f"  Sent 'I am going offline.' to room {token}")
        await nextcloud.stop()
        await stdout_channel.stop()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
