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
        Configured ToolManager with get_weather and calculate tools.
    """
    tool_manager = ToolManager()

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny and 25°C in {city}"

    def calculate(expression: str) -> str:
        """Calculate a simple arithmetic expression safely."""
        try:
            import ast
            import operator

            op_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.Mod: operator.mod,
            }

            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    left = eval_expr(node.left)
                    right = eval_expr(node.right)
                    return op_map[type(node.op)](left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = eval_expr(node.operand)
                    if isinstance(node.op, ast.USub):
                        return -operand
                    return operand
                else:
                    raise ValueError("Unsupported expression")

            tree = ast.parse(expression, mode="eval")
            result = eval_expr(tree.body)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    tool_manager.register_tool(func=get_weather)
    tool_manager.register_tool(func=calculate)

    return tool_manager


async def main():
    """Main entry point."""
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

    # Start the Agent's event loop
    await agent.start()
    print("Agent event loop started")
    print()

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

    # Run the shell interactively (blocks until /quit)
    await shell.run()

    print()
    print("Goodbye!")

    # Cleanup
    await agent.stop()
    print("Agent event loop stopped")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
