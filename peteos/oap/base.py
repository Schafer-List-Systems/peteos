"""AgenticObjectBase - base class for all OAP objects."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any

from peteos.chatbot import ContentPart, Message
from peteos.executionenvironment import ExecStatus
from peteos.oap.error import Error
from peteos.role import Role
from peteos.toolmanager import Tool, ToolManager

from peteos.oap.decorators import tool

if TYPE_CHECKING:
    from peteos.agent import Agent
    from peteos.session import Session


class AgenticObjectBase:
    """Base class for all Object-Agentic Programming objects."""

    def __init__(self) -> None:
        self._oap_agent: Agent | None = None
        self._oap_role: Role = self._create_role()
        self._oap_lock: threading.Lock = threading.Lock()
        self._oap_tool_manager: ToolManager = ToolManager()
        self._oap_current_output_schema: type | None = None
        self._register_tools()
        self._register_output_schema_hook()

    def _create_role(self) -> Role:
        """Create the Role for this object."""
        class_name = self.__class__.__name__
        return Role(
            name=f"oap_{class_name}",
            description=f"Agent for {class_name}",
        )

    def _register_tools(self) -> None:
        """Register @tool-decorated methods from this class and its parents."""
        registered: set[str] = set()
        for cls in self.__class__.__mro__:
            for name, method in cls.__dict__.items():
                if callable(method) and hasattr(method, "_tool_name"):
                    tool_name = method._tool_name
                    if tool_name not in registered:
                        registered.add(tool_name)
                        t = Tool(
                            name=tool_name,
                            description=method._tool_description or "",
                            func=getattr(self, name),
                        )
                        self._oap_tool_manager.register_tool(t)

    def _register_output_schema_hook(self) -> None:
        """Register the output schema system prompt hook."""
        self._oap_role.add_system_prompt_hook(self._output_schema_hook)

    def _output_schema_hook(self) -> str:
        """System prompt hook: returns formatted output schema description."""
        if self._oap_current_output_schema is None:
            return ""
        schema_name = self._oap_current_output_schema.__name__
        return f"\n\n# Output Schema\nYour final answer must be produced via produce_output with an object matching {schema_name}."

    @tool(name="produce_output", description="Signal your final answer. Pass the result as a JSON-compatible value (str, int, float, bool, list, or dict).")
    def _produce_output(self, data: Any, session: "Session | None" = None) -> str:
        """Protected tool: signals the agent has produced its final answer.

        Validates the data against the current output schema (if set),
        writes the result to the session's AgenticState, and returns
        the validation result. A non-OK response serves as error
        feedback the agent can use to correct its tool call.

        Args:
            data: The result to produce (any JSON-compatible value).
            session: The session (injected by the execution environment).

        Returns:
            "OK" on success, or an error message the agent can fix.
        """
        if session is None:
            return "Error: session not available."
        if self._oap_current_output_schema is not None:
            error = self._validate_produced_data(data)
            if error:
                return error
        try:
            session.state.create("_oap_produced_data", json.dumps(data))
        except ValueError as e:
            return f"Error: {e}"
        return "OK"

    def _validate_produced_data(self, data: Any) -> str | None:
        """Validate data against the current output schema.

        Returns an error message string if validation fails, None if valid.
        """
        schema = self._oap_current_output_schema
        if schema is str:
            if not isinstance(data, str):
                return "Error: expected str, got " + type(data).__name__
            return None
        if schema is int:
            if not isinstance(data, int) or isinstance(data, bool):
                return "Error: expected int, got " + type(data).__name__
            return None
        if schema is float:
            if not isinstance(data, (int, float)) or isinstance(data, bool):
                return "Error: expected float, got " + type(data).__name__
            return None
        if schema is bool:
            if not isinstance(data, bool):
                return "Error: expected bool, got " + type(data).__name__
            return None
        if schema is list:
            if not isinstance(data, list):
                return "Error: expected list, got " + type(data).__name__
            return None
        if schema is dict:
            if not isinstance(data, dict):
                return "Error: expected dict, got " + type(data).__name__
            return None
        if schema is str | int | float | bool | list | dict | type(None):
            # Union type — allow any JSON-compatible value
            return None
        return None

    def acquire(self, timeout: float | None = None) -> None:
        """Acquire the invocation lock.

        Blocks until the lock is available or timeout expires.
        The lock serializes concurrent `invoke()` calls on the same object
        to prevent race conditions from interleaved @tool method calls.

        Args:
            timeout: Maximum seconds to wait. `None` = block indefinitely.

        Raises:
            TimeoutError: Lock not acquired within timeout.
        """
        if timeout is None:
            self._oap_lock.acquire()
        elif not self._oap_lock.acquire(timeout=timeout):
            raise TimeoutError(
                f"Could not acquire invocation lock within {timeout}s"
            )

    def release(self) -> None:
        """Release the invocation lock.

        Must be called exactly once for each successful `acquire()`.
        Called via try/finally in `invoke()` to guarantee release even on error.

        Raises:
            RuntimeError: Lock is not held by the calling thread.
        """
        self._oap_lock.release()

    def update_system_prompt(self, prompt: str) -> None:
        """Update the agent's system prompt.

        Updates the `_oap_role` with the new prompt so future invocations
        use the updated system prompt.

        Args:
            prompt: New system prompt text.
        """
        if self._oap_role is not None:
            self._oap_role.system_prompt = prompt

    @property
    def agent(self) -> Agent | None:
        """The Agent instance used for LLM invocations."""
        return self._oap_agent

    @agent.setter
    def agent(self, value: Agent | None) -> None:
        self._oap_agent = value

    @property
    def role(self) -> Role:
        """The Role used for this object's invocations."""
        return self._oap_role

    async def invoke_agent(
        self,
        prompt: str,
        output_schema: type | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Invoke this object's agent.

        Acquires the invocation lock, creates a Session, queues the prompt,
        waits for produce_output (via poll loop on session state), extracts
        structured output, then releases the lock.

        Args:
            prompt: Task description for the agent.
            output_schema: Expected return type (dataclass, etc.).
            timeout: Maximum seconds to wait for the invocation lock.

        Returns:
            Structured output, Error object, or raises Exception.

        Raises:
            ValueError: No _oap_agent set.
            TimeoutError: Lock not acquired within timeout.
        """
        # --- Gatekeeper checks ---
        if self._oap_agent is None:
            raise ValueError("No Agent available")

        # --- Acquire lock with timeout ---
        self.acquire(timeout)

        # --- Update output schema (serialized by lock) ---
        self._oap_current_output_schema = output_schema

        # --- Create Session via self's agent (canonical peteos approach) ---
        session: Session = await self._oap_agent.create_session()

        try:
            # --- Wait for produce_output via after_step hook ---
            done = asyncio.Event()
            reminder_msg = "Please produce your final output now using produce_output."
            start_time = time.time()

            def _on_step_done(sess: "Session", status: ExecStatus) -> None:
                if sess.state._data.get("_oap_produced_data") is not None:
                    done.set()
                    return
                elapsed = time.time() - start_time
                if timeout is not None and elapsed > timeout:
                    done.set()  # signal timeout to main loop
                    return
                sess.push_event(Message(
                    role="user",
                    content=[ContentPart(part_type="text", text=reminder_msg)],
                ))

            session.execution_environment.register_hook(
                "after_step", _on_step_done, session
            )

            try:
                await asyncio.wait_for(done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return Error("Agent did not produce output within timeout")

            # --- Extract and parse structured output ---
            produced_data_str = session.state._data.get("_oap_produced_data")
            if produced_data_str is None:
                return Error("Agent did not produce output within timeout")
            produced_data = json.loads(produced_data_str)
            if output_schema is not None and is_dataclass(output_schema):
                produced_data = output_schema(**produced_data)

            return produced_data
        finally:
            try:
                await session.stop()
            except Exception:
                pass
            self.release()

    async def invoke(
        self,
        target: "AgenticObjectBase",
        prompt: str,
        output_schema: type | None = None,
        persistent: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """Invoke a sub-agent on a target AgenticObjectBase.

        Verifies that self allows sub-agent invocation, then forwards to
        target.invoke_agent().

        Args:
            target: The sub-object to invoke the sub-agent on.
            prompt: Task description for the sub-agent.
            output_schema: Expected return type (dataclass, etc.).
            persistent: If True, inherit the parent's thread ID.
            timeout: Maximum seconds to wait for the invocation lock on the target.

        Returns:
            Structured output, Error object, or raises Exception.

        Raises:
            ValueError: Target has no _oap_agent set.
            TimeoutError: Lock not acquired within timeout.
        """
        # --- Gatekeeper: verify self allows sub-agent invocation ---
        config = getattr(self, "_oap_config", {})
        if not config.get("invoke_sub_agents", False):
            return Error("Sub-agent invocation not enabled")

        # --- Forward to target's invoke_agent ---
        return await target.invoke_agent(
            prompt=prompt,
            output_schema=output_schema,
            timeout=timeout,
        )
