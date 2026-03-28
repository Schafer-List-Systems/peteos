import json
import re
from typing import Any, Dict, List

from peteos.chatbot import ChatBot
from peteos.chathistory import ChatHistory
from peteos.executionenvironment import ExecutionEnvironment
from peteos.message import Message
from peteos.toolmanager import ToolManager


class REPLExecutionEnvironment(ExecutionEnvironment):
    """REPL (Read-Eval-Print Loop) execution environment."""

    def __init__(
        self,
        chatbot: ChatBot,
        chat_history: ChatHistory,
        tool_manager: ToolManager
    ):
        """
        Initialize REPLExecutionEnvironment.

        Args:
            chatbot: The ChatBot instance to use.
            chat_history: The ChatHistory instance to use.
            tool_manager: The ToolManager instance to use.
        """
        super().__init__(chatbot=chatbot, chat_history=chat_history, tool_manager=tool_manager)
        self._interrupt = False

    async def run(self) -> None:
        """
        Run the REPL loop.

        Reads input, processes it through the chatbot, and appends output to ChatHistory.
        Loops until final answer is received or interrupt flag is set.
        """
        while not self._interrupt:
            # Send chat history to chatbot
            response = await self.chatbot.send_message(
                self.chat_history,
                streaming=True
            )

            # Collect accumulated response
            async for _ in response:
                # Streaming - just wait for completion
                pass

            # Check if we're interrupted
            if self._interrupt:
                break

            # Extract thinking and text content
            thinking_content = response.thinking_content
            text_content = response.text_content

            # Append thinking content if present
            if thinking_content:
                self.chat_history.append_message(
                    Message(content={
                        "role": "assistant",
                        "content": f"[Thinking]\n{thinking_content}"
                    })
                )

            # Parse response for tool calls
            tool_calls = self._parse_tool_calls(text_content)

            if tool_calls:
                # Append the response containing tool calls as assistant message
                self.chat_history.append_message(
                    Message(content={
                        "role": "assistant",
                        "content": text_content
                    })
                )
                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    args = tool_call.get("arguments", {})

                    tool = self.tool_manager.get_tool(tool_name)
                    if tool:
                        result = tool.execute(**args)
                        # Append tool result
                        self.chat_history.append_message(
                            Message(content={
                                "role": "tool",
                                "name": tool_name,
                                "content": str(result)
                            })
                        )
                # Loop continues - sends history with tool results back to LLM
            else:
                # Final answer - append and exit loop
                self.chat_history.append_message(
                    Message(content={
                        "role": "assistant",
                        "content": text_content
                    })
                )
                break

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from response content.

        Looks for OpenAI-style tool_calls JSON object.

        Args:
            content: Response text content to parse.

        Returns:
            List of tool call dicts with name and arguments.
        """
        # Try direct JSON object without code fence
        try:
            obj = json.loads(content.strip())
            if isinstance(obj, dict) and "name" in obj:
                return [obj]
            elif isinstance(obj, dict) and "tool_calls" in obj:
                return obj["tool_calls"]
            elif isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

        # Find JSON object with nested braces
        start = content.find('{')
        if start != -1:
            brace_count = 0
            end = start
            for i, c in enumerate(content[start:]):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i
                        break

            json_str = content[start:end + 1]
            try:
                tool_call = json.loads(json_str)
                if isinstance(tool_call, dict) and "name" in tool_call:
                    return [tool_call]
            except json.JSONDecodeError:
                pass

        # Search for tool_calls JSON pattern in code fences
        pattern = r'```(?:tool_calls)?\n?\s*(\{[^}]+\})\s*```'
        match = re.search(pattern, content)

        if match:
            try:
                tool_call_json = match.group(1)
                tool_call = json.loads(tool_call_json)
                # Normalize to list format
                if isinstance(tool_call, dict) and "name" in tool_call:
                    return [tool_call]
                elif isinstance(tool_call, dict) and "tool_calls" in tool_call:
                    return tool_call["tool_calls"]
            except json.JSONDecodeError:
                pass

        return []

    def set_interrupt(self) -> None:
        """Set the interrupt flag to request loop termination."""
        self._interrupt = True

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        self._interrupt = False
