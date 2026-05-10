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


def load_nextcloud_config(config_file: str = "examples/config/nextcloud_config.json"):
    """Load Nextcloud Talk bot configuration from a JSON file.

    Expected format:
    {
        "nextcloud_url": "https://cloud.example.com",
        "bot_id": "mybot",
        "bot_secret": "your-shared-secret",
        "default_role": "test",
        "host": "0.0.0.0",
        "port": 8766
    }

    IMPORTANT: Store secrets in examples/config/nextcloud_config.json and
    add that file to .gitignore. Never commit secrets to version control.

    Also required: enable features after bot installation:
    ./occ talk:bot:state <bot-id> 1 --feature webhook --feature response --feature reaction

    Args:
        config_file: Path to the JSON configuration file.

    Returns:
        Dictionary with Nextcloud configuration values.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        KeyError: If required fields are missing.
    """
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found.")
        print("Create it from the template:")
        print(f"  cp examples/config/nextcloud_config.json.example {config_file}")
        raise

    required = ["nextcloud_url", "bot_id", "bot_secret"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Missing required config fields: {', '.join(missing)}")

    return config


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
    role_manager = setup_role_manager()
    chatbot_manager = await setup_chatbot_manager(args.chatbot_config)
    tool_manager = setup_tool_manager()

    # Create the Agent
    agent = Agent(role_manager, chatbot_manager, tool_manager)

    # Load Nextcloud configuration
    try:
        nc_config = load_nextcloud_config(args.nextcloud_config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Aborting: {e}")
        return

    # Create the Nextcloud Talk channel
    nextcloud = NextcloudTalkChannel(
        name="nextcloud",
        agent=agent,
        nextcloud_url=nc_config["nextcloud_url"],
        bot_id=nc_config["bot_id"],
        bot_secret=nc_config["bot_secret"],
        default_role=nc_config.get("default_role", "test"),
        host=nc_config.get("host", "0.0.0.0"),
        port=nc_config.get("port", 8766),
    )

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
    print("Sending initial status message to active conversation...")
    # Send startup message to the most recent conversation if one exists
    if nextcloud._rooms:
        token = list(nextcloud._rooms.keys())[-1]
        await nextcloud._send_to_nextcloud(token, "Hello, I am online now.")
        print(f"  Sent 'Hello, I am online now.' to room {token}")
    else:
        print("  No active rooms yet. Will greet on first room join.")
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
            for token in nextcloud._rooms:
                await nextcloud._send_to_nextcloud(token, "I am going offline.")
        await nextcloud.stop()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
