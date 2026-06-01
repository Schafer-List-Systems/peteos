"""AgenticObjectBase - base class for all OAP objects."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any

from peteos.chatbot import ContentPart, Message
from peteos.executionenvironment import ExecStatus
from peteos.logger import get_logger
from peteos.oap.error import Error
from peteos.oap.sandbox import create_sandbox_globals
from peteos.role import Role
from peteos.toolmanager import Tool, ToolManager

from peteos.oap.decorators import tool

_logger = get_logger(__name__)

if TYPE_CHECKING:
    from peteos.session import Session

from peteos.agent import Agent


class AgenticObjectBase:
    """Base class for all Object-Agentic Programming objects."""

    def __init__(self) -> None:
        self._oap_role: Role = self._create_role()
        self._oap_lock: threading.Lock = threading.Lock()
        self._oap_tool_manager: ToolManager = ToolManager()
        self._oap_current_output_schema: type | None = None
        self._oap_thread_store: dict[str, UUID] = {}
        self._register_tools()
        self._register_output_schema_hook()
        self._register_sandbox_tool()
        self._oap_agent: Agent = self._create_agent()
        tool_names = [t.name for t in self._oap_tool_manager.get_tool_list()]
        _logger.debug("Registered tools for %s: %s", self.__class__.__name__, tool_names)

    def _create_role(self) -> Role:
        """Create the Role for this object."""
        class_name = self.__class__.__name__
        return Role(
            name=f"oap_{class_name}",
            description=f"Agent for {class_name}",
            tool_filter=[".*"],
            system_prompt=(
                f"You are an agent working on a {class_name} object. "
                f"You have tools to read and modify the state of this object. "
                f"Use those tools and the information you already have to fulfill the user's request. "
                f"NEVER ask the user for more information or clarification. "
                f"If you cannot produce the requested output, use produce_output to return an error message explaining why. "
            ),
        )

    def _create_agent(self) -> Agent:
        """Create an Agent wired to this object's role and tool_manager."""
        # Auto-approve all registered tools so they pass Session.add_tool_call()
        for t in self._oap_tool_manager.get_tool_list():
            self._oap_role.auto_approve_tools.append(t.name)
        return Agent(self._oap_role, self._oap_tool_manager)

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
        if is_dataclass(self._oap_current_output_schema):
            import dataclasses
            fields = dataclasses.fields(self._oap_current_output_schema)
            field_lines = "\n".join(f"  - {f.name}: {f.type.__name__}" for f in fields)
            return f"\n\n# Output Schema\nYour final answer must be produced via produce_output with a JSON object containing these fields:\n{field_lines}."
        return f"\n\n# Output Schema\nYour final answer must be produced via produce_output with a value of type {schema_name}."

    def _register_sandbox_tool(self) -> None:
        """Register python_exec tool when allow_code_execution is enabled on the class."""
        config = getattr(self.__class__, "_oap_config", {})
        if not config.get("allow_code_execution", False):
            return
        self._oap_tool_manager.register_tool(
            Tool(
                name="python_exec",
                description="Execute sandboxed Python code. Access the object graph via `self`. No __builtins__, no __import__, no network, no filesystem.",
                func=self._python_exec,
            )
        )

    def _python_exec(self, code: str) -> str:
        """Protected tool: executes sandboxed Python code."""
        config = getattr(self.__class__, "_oap_config", {})
        imports = config.get("imports", [])
        sandbox_globals = create_sandbox_globals(self, imports)

        try:
            exec(code, sandbox_globals)
            return "OK"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool(name="produce_output", description="Signal your final answer. Pass the result as a JSON string describing the output data.")
    def _produce_output(self, data: str, session: "Session | None" = None) -> str:
        """Protected tool: signals the agent has produced its final answer.

        Parses the JSON string and validates it against the current output
        schema. If validation fails, returns an error message the agent can
        use to correct its tool call.

        Args:
            data: The result as a JSON string.
            session: The session (injected by the execution environment).

        Returns:
            "OK" on success, or an error message the agent can fix.
        """
        if session is None:
            return "Error: session not available."
        # Parse JSON — the LLM always sends a string.
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON: {e}"

        # Validate and cast to schema if set.
        error = self._validate_produced_data(parsed)
        if error:
            return error

        # Cast to dataclass if applicable.
        if self._oap_current_output_schema is not None and is_dataclass(self._oap_current_output_schema):
            if isinstance(parsed, dict):
                parsed = self._oap_current_output_schema(**parsed)

        try:
            session.state.create("_oap_produced_data", parsed)
        except ValueError as e:
            return f"Error: {e}"
        return "OK"

    @tool(name="produce_error", description="Signal that you could not produce the requested output. Pass an error message explaining why (e.g., missing required data or an invalid state).")
    def _produce_error(self, message: str, session: "Session | None" = None) -> str:
        """Protected tool: signals the agent could not fulfill the task.

        Writes the error message to the session's AgenticState so that
        invoke_agent returns an Error object.

        Args:
            message: Human-readable error explanation.
            session: The session (injected by the execution environment).

        Returns:
            "OK" on success, or an error message if the session is missing.
        """
        if session is None:
            return "Error: session not available."
        try:
            session.state.create("_oap_error", message)
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
        persistent_thread_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Invoke this object's agent.

        Acquires the invocation lock, creates or reuses a Session, queues the
        prompt, waits for produce_output (via after_step hook), extracts
        structured output, then releases the lock.

        Args:
            prompt: Task description for the agent.
            output_schema: Expected return type (dataclass, etc.).
            persistent_thread_id: If set, reuses or creates a persistent
                session keyed by this ID. If None, creates a transient
                session destroyed after the invocation.
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

        # --- Create or reuse session ---
        session: Session | None = None
        if persistent_thread_id is not None:
            stored_uuid = self._oap_thread_store.get(persistent_thread_id)
            if stored_uuid is not None:
                session = self._oap_agent.get_session(stored_uuid)
        if session is None:
            session = await self._oap_agent.create_session()
            if persistent_thread_id is not None:
                self._oap_thread_store[persistent_thread_id] = session.uuid

        try:
            # --- Wait for produce_output via after_step hook ---
            done = asyncio.Event()
            reminder_msg = "Please produce your final output using produce_output() or produce_error(). Mind the output schema!"
            start_time = time.time()

            async def _on_step_done(sess: "Session", status: ExecStatus) -> None:
                if sess.state._data.get("_oap_produced_data") is not None:
                    done.set()
                    sess.execution_environment.set_interrupt()
                    return
                if sess.state._data.get("_oap_error") is not None:
                    done.set()
                    sess.execution_environment.set_interrupt()
                    return
                elapsed = time.time() - start_time
                if timeout is not None and elapsed > timeout:
                    done.set()
                    sess.execution_environment.set_interrupt()
                    return
                await sess.queue_message(Message(
                    role="user",
                    content=[ContentPart(part_type="text", text=reminder_msg)],
                ))

            session.execution_environment.register_hook(
                "after_step", _on_step_done, session
            )

            # --- Queue the initial prompt so the agent actually runs ---
            await session.queue_message(Message(
                role="user",
                content=[ContentPart(part_type="text", text=prompt)],
            ))

            try:
                await asyncio.wait_for(done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return Error("Agent did not produce output within timeout")

            # --- Extract and parse structured output ---
            produced_data = session.state._data.get("_oap_produced_data")
            if produced_data is not None:
                return produced_data

            error_msg = session.state._data.get("_oap_error")
            if error_msg is not None:
                return Error(error_msg)

            return Error("Agent did not produce output within timeout")
        finally:
            # Clear OAP state variables so the next invocation starts fresh
            if session is not None:
                try:
                    session.state._data.pop("_oap_produced_data", None)
                    session.state._data.pop("_oap_error", None)
                except Exception:
                    pass
            if persistent_thread_id is None:
                try:
                    await session.stop()
                    self._oap_agent.destroy_session(session.uuid)
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
