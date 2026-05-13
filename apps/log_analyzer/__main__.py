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
import json
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

from apps.log_analyzer.tools import _state, register_filter_tools, register_state_tools


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
    tm = ToolManager()

    # Create the Agent
    agent = Agent(role_manager, chatbot_manager, tm)

    # Register awakeness status hook on the role's system prompt
    role_name = nextcloud_config.get("default_role", "router")
    role = role_manager.get_role(role_name)
    if role is not None:
        role.add_system_prompt_hook(
            lambda: "Status: you are currently silent.\n" if (_state.nextcloud_channel and _state.nextcloud_channel.is_session_silent(_state.router_session_uuid)) else "Status: you are currently verbose.\n"
        )

    # Create the stdout channel first so its methods are available for tool registration
    # (tools access the channel via _state.channel, not the tool manager)
    stdout_channel = ReadStdoutChannel(
        name="log-monitor",
        agent=agent,
        config=stdout_config,
    )

    # Register all tools (before session creation so chat history includes them)
    register_state_tools(tm)
    register_filter_tools(tm, stdout_channel)

    # Create a shared session that both channels attach to
    session = await agent.create_session(role_name)

    # Register rolling window discard hook
    try:
        with open("apps/log_analyzer/config/app_config.json", "r") as f:
            app_cfg = json.load(f)
        max_tokens = app_cfg.get("rolling_window_max_tokens", 60000)
    except FileNotFoundError:
        max_tokens = 60000

    async def _on_before_loop_continue(_delta_messages, _max=max_tokens):
        _, total_tokens = session.execution_environment.chat_history.rolling_window_discard(_max)
        _state.last_context_tokens = total_tokens
        return None

    session.execution_environment.register_hook("before_loop_continue", _on_before_loop_continue)

    print(f"Created session: {session.uuid}")
    print()

    # Create the Nextcloud Talk channel (user-facing, sends and receives)
    nextcloud = NextcloudTalkChannel(name="nextcloud", agent=agent, config=nextcloud_config)

    # Wire the nextcloud channel to the tools for per-session silent mode
    _state.set_nextcloud_channel(nextcloud)
    _state.router_session_uuid = session.uuid

    # Add context size awareness to the system prompt (reads cached count from _state)
    from peteos.chatbot.message import SystemPromptMessage
    for msg in session.chat_history.messages:
        if isinstance(msg, SystemPromptMessage):
            msg.add_hook(lambda: f"Context: {_state.last_context_tokens} of {max_tokens} tokens used.\n")
            break

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
