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


from peteos.oap._schema import cast_produced_data, get_schema_description, validate_produced_data


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
        cls = self.__class__
        class_name = cls.__name__
        doc = (cls.__doc__ or "").strip() if cls is not AgenticObjectBase else ""
        if doc:
            system_prompt = doc
        else:
            system_prompt = (
                f"You are an agent working on a {class_name} object. "
                f"You have tools to read and modify the state of this object. "
                f"Use those tools and the information you already have to fulfill the user's request. "
                f"NEVER ask the user for more information or clarification. "
                f"If you cannot produce the requested output, use produce_output to return an error message explaining why. "
            )
        return Role(
            name=f"oap_{class_name}",
            description=f"Agent for {class_name}",
            tool_filter=[".*"],
            system_prompt=system_prompt,
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
        desc = get_schema_description(self._oap_current_output_schema)
        if desc is not None:
            json_schema, docstring = desc
            return f"# Output\n\nProvide your final answer with the `produce_output` tool with a JSON object matching:\n{json_schema}\nSchema description: {docstring}"
        if self._oap_current_output_schema is not None:
            return f"# Output\n\nProvide your final answer with the `produce_output` tool with a value of type {self._oap_current_output_schema.__name__}."
        return (
            "# Output\n\nProvide your final answer with the `produce_output` tool. "
            "Any value (string, array, or dict) is acceptable."
        )

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

    def _register_media_tool(self) -> None:
        """Register read_media tool when allow_media_access is enabled on the class."""
        config = getattr(self.__class__, "_oap_config", {})
        if not config.get("allow_media_access", False):
            return
        self._oap_tool_manager.register_tool(
            Tool(
                name="read_media",
                description="Read a local media file (image, video, PDF) or fetch one from a URL. The content is provided as user message.",
                func=self._read_media,
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

    @tool(name="produce_output", description="Produce the desired output and signal your final answer. Pass the result as a JSON string describing the output data.")
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
            _logger.debug("_produce_output: session is None")
            return "Error: session not available."
        # Parse JSON — the LLM always sends a string.
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            schema_hint = ""
            desc = get_schema_description(self._oap_current_output_schema)
            if desc is not None:
                json_schema, docstring = desc
                schema_hint = f"Provide your final answer with the `produce_output` tool with a JSON object matching:\n{json_schema}\nSchema description: {docstring}"
            return f"Error: invalid JSON: {e}\n{schema_hint}"

        # Validate and cast to schema if set.
        error = validate_produced_data(parsed, self._oap_current_output_schema)
        if error:
            return error

        parsed = cast_produced_data(parsed, self._oap_current_output_schema)

        try:
            session.state.create("_oap_produced_data", parsed)
            _logger.debug(
                "_produce_output: wrote to session %s, _oap_produced_data=%s",
                session.uuid, parsed,
            )
        except ValueError as e:
            return f"Error: {e}"
        return "OK"

    @tool(
        name="read_media",
        description="Read a local media file (image, video, PDF) or fetch one from a URL. The content is provided as user message.",
    )
    async def _read_media(self, src: str, session: "Session | None" = None) -> str:
        """Tool: load a media file and queue it back to the agent's session.

        Reads the file from disk or fetches from a URL, encodes it as a
        ContentPart (image/video/pdf), and queues a new user message into
        the session's event queue. The session loop will drain this message
        and add it to chat history before the next LLM call.

        Args:
            src: Local file path or HTTP(S) URL to the media file.
            session: The session (injected by the execution environment).

        Returns:
            Confirmation message with file info.
        """
        from peteos.chatbot import ContentPart, Message

        if session is None:
            return "Error: session not available."

        from peteos.utils.image import create_content_part_async
        media_part = await create_content_part_async(src, timeout=30.0)

        queued_msg = Message(
            role="user",
            content=[
                ContentPart(part_type="text", text=f"Media loaded from {src}."),
                media_part,
            ],
        )
        await session.queue_message(queued_msg)
        _logger.debug("_read_media: queued media from %s for session %s", src, session.uuid)
        return f"OK: media queued from {src}"

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
            _logger.error(
                "Could not acquire invocation lock for %s within %.1fs",
                self.__class__.__name__,
                timeout,
            )
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
        image: str | None = None,
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
            image: Optional local file path or HTTP(S) URL to attach an image
                to the prompt. The image is base64-encoded and sent alongside
                the text prompt.

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
            async def _on_step_done(sess: "Session", status: ExecStatus) -> ExecStatus | None:
                produced = sess.state._data.get("_oap_produced_data")
                errored = sess.state._data.get("_oap_error")
                if errored is not None:
                    _logger.debug("_on_step_done: error found, returning FINISHED")
                    return ExecStatus.FINISHED
                elif produced is not None:
                    _logger.debug("_on_step_done: produced data found, returning FINISHED")
                    return ExecStatus.FINISHED
                else:
                    schema_desc = ""
                    desc = get_schema_description(self._oap_current_output_schema)
                    if desc is not None:
                        json_schema, docstring = desc
                        schema_desc = (
                            f"Provide your final answer with the `produce_output` tool with a JSON object matching:\n{json_schema}\nSchema description: {docstring}"
                        )
                    elif self._oap_current_output_schema is not None:
                        schema_desc = (
                            f"Provide your final answer with the `produce_output` tool with a value of type {self._oap_current_output_schema.__name__}."
                        )
                    reminder = (
                        f"It looks like you finished a step without calling `produce_output` or `produce_error`. "
                        f"If you have your final answer, call `produce_output` with your result!{schema_desc} "
                    )
                    await session.queue_message(Message(
                        role="user",
                        content=[ContentPart(part_type="text", text=reminder)],
                    ))
                    return None

            session.execution_environment.register_hook(
                "after_step", _on_step_done, session
            )

            content: list[ContentPart] = [ContentPart(part_type="text", text=prompt)]
            if image is not None:
                from peteos.utils.image import create_image_content_part_async
                image_part = await create_image_content_part_async(image, timeout or 30.0)
                content.append(image_part)

            await session.queue_message(Message(
                role="user",
                content=content,
            ))

            reminder_msg = f"Please produce your final output using the `produce_output` or `produce_error` tool.\n" \
                           f"Mind the output schema!"
            start_time = time.time()

            while True:
                await session.wait_for_idle(timeout=timeout)
                produced_data = session.state._data.get("_oap_produced_data")
                if produced_data is not None:
                    return produced_data
                error_msg = session.state._data.get("_oap_error")
                if error_msg is not None:
                    return Error(error_msg)

                elapsed = time.time() - start_time
                if timeout is not None and elapsed > timeout:
                    _logger.error(
                        "invoke_agent timeout hit for %s thread=%s after %.1fs",
                        self.__class__.__name__,
                        persistent_thread_id,
                        elapsed,
                    )
                    return Error(f"Agent did not produce output within {timeout}s timeout")

                await session.queue_message(Message(
                    role="user",
                    content=[ContentPart(part_type="text", text=reminder_msg)],
                ))
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
ch habe noch...
