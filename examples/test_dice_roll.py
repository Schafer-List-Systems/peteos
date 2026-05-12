#!/usr/bin/env python3
"""Test dice roll tool call loop."""

import asyncio
from peteos.agent import Agent
from peteos.chatbot.manager import ChatBotManager
from peteos.logger import setup_logging
from peteos.role import Role
from peteos.rolemanager import RoleManager
from peteos.toolmanager import ToolManager, Tool
from peteos.chatbot.message import Message
from peteos.chatbot.contentpart import ContentPart

async def test():
    setup_logging(level='DEBUG', debug=True)

    role_manager = RoleManager()
    role_manager.register_role(Role(name='test', description='Test role', model='.*'))

    chatbot_manager = ChatBotManager()
    await chatbot_manager.load_from_file('config/chatbot_config.json')

    tool_manager = ToolManager()

    def random_number(min_val=1, max_val=10):
        """Generate a random number between min_val and max_val (inclusive)."""
        import random
        return random.randint(min_val, max_val)

    tool_manager.register_tool(Tool.from_callable(random_number))

    agent = Agent(role_manager, chatbot_manager, tool_manager)
    await agent.start()

    session = agent.create_session('test')
    print(f'Session created: {session.uuid}')
    print()

    # Send message to roll dice
    msg = Message(
        role='user',
        content=[ContentPart(part_type='text', text='Roll a dice')]
    )
    agent.post_message(session.uuid, msg)
    print('Posted: "Roll a dice"')
    print()

    # Wait for response
    await asyncio.sleep(10)

    # Show messages
    print('=== Messages ===')
    for i, msg in enumerate(session.chat_history.messages):
        print(f'{i}. Role: {msg.get_role()}')
        print(f'   Text: {msg.text[:200]}...')
        print()

    await agent.stop()
    print('Done!')

if __name__ == '__main__':
    asyncio.run(test())
