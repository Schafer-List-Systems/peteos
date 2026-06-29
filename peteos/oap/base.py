"""AgenticObjectBase - base class for all OAP objects."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, is_dataclass
from typing import Any

from peteos.conversation import ContentPart, Message
from peteos.conversation.session import Session
from peteos.engine import ExecStatus, Runner
from peteos.persona.role import Role
from peteos.persona.toolmanager import Tool, ToolManager
from peteos.utils import get_logger
from peteos.oap.error import Error
from peteos.oap.sandbox import build_sandbox_description, create_sandbox_globals

from peteos.oap.decorators import tool
from peteos.oap.agentic_registry import AgenticObjectRegistry

_logger = get_logger(__name__)

_AGENT_BASE_DIR = os.environ.get("PETEOS_AGENT_BASE_DIR", "/tmp/peteos")


def _build_system_prompt(cls: type) -> str:
    """Build the system prompt for a concrete agentic object class.

    Collects docstrings from classes in the MRO that directly inherit
    from AgenticObjectBase (concrete agentic objects), ordered from
    most-derived to base. Prepends standard behaviour directives.
    """
    class_name = cls.__name__

    # Collect only concrete agentic objects — classes that directly
    # inherit from AgenticObjectBase — in MRO order (derived first).
    doc_parts: list[str] = []
    for parent in cls.__mro__:
        if parent in (AgenticObjectBase, object):
            continue
        if AgenticObjectBase not in parent.__bases__:
            continue
        parent_doc = (parent.__doc__ or "").strip()
        if parent_doc:
            doc_parts.append(parent_doc)

    if doc_parts:
        # Return the combined class descriptions from the inheritance chain.
        return "\n\n".join(doc_parts)
    return (
        f"You are an agent working on a {class_name} object. "
        f"You have tools to read and modify the state of this object. "
        f"Use those tools and the information you already have to fulfill the user's request. "
        f"NEVER ask the user for more information or clarification. "
        f"If you cannot produce the requested output, use the `produce_error` to return an error message explaining why. "
    )


def _collect_oap_config(cls: type) -> dict[str, Any]:
    """Collect and merge OAP config from all classes in the MRO that directly inherit from AgenticObjectBase.

    Mirrors the MRO iteration in _build_system_prompt. Collects the union of
    all imports, merges import_aliases, and ORs all boolean flags across the
    diamond hierarchy so that a class D(B, C) where both B and C define
    @agentic_object with different imports gets all of them combined.
    """
    imports: set[object] = set()
    import_aliases: dict[str, str] = {}
    allow_code_execution = False
    allow_media_access = False
    invoke_sub_agents = False
    for parent in cls.__mro__:
        if parent in (AgenticObjectBase, object):
            continue
        if AgenticObjectBase not in parent.__bases__:
            continue
        cfg = getattr(parent, "_oap_config", None)
        if cfg:
            allow_code_execution |= cfg.get("allow_code_execution", False)
            allow_media_access |= cfg.get("allow_media_access", False)
            invoke_sub_agents |= cfg.get("invoke_sub_agents", False)
            imports.update(cfg.get("imports", []))
            import_aliases.update(cfg.get("import_aliases", {}))
    return {
        "allow_code_execution": allow_code_execution,
        "allow_media_access": allow_media_access,
        "invoke_sub_agents": invoke_sub_agents,
        "imports": list(imports),
        "import_aliases": import_aliases,
    }


from peteos.oap._schema import get_schema_description, parse_data


from peteos.persona.agent import Agent


class AgenticObjectBase:
    """Base class for all Object-Agentic Programming objects.

    All subclasses are auto-registered in AgenticObjectRegistry.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        AgenticObjectRegistry.register(cls)

    def __init__(self) -> None:
        super().__init__()
        self._oap_role: Role = self._create_role()
        self._oap_lock: threading.Lock = threading.Lock()
        self._oap_tool_manager: ToolManager = ToolManager()
        self._oap_current_output_schema: type | None = None
        self._oap_thread_store: dict[str, str] = {}
        self._register_tools()
        self._register_output_schema_hook()
        self._register_sandbox_tool()
        self._register_media_tool()
        self._oap_agent: Agent = self._create_agent()
        tool_names = [t.name for t in self._oap_tool_manager.get_tool_list()]
        _logger.debug("Registered tools for %s: %s", self.__class__.__name__, tool_names)

    def _create_role(self) -> Role:
        """Create the Role for this object."""
        cls = self.__class__
        system_prompt = _build_system_prompt(cls)
        return Role(
            name=f"oap_{cls.__name__}",
            description=f"Agent for {cls.__name__}",
            tool_filter=[".*"],
            system_prompt=system_prompt,
        )

    def _create_agent(self) -> Agent:
        """Create an Agent wired to this object's role and tool_manager."""
        # Auto-approve all registered tools so they pass Runner.add_tool_call()
        for t in self._oap_tool_manager.get_tool_list():
            self._oap_role.auto_approve_tools.append(t.name)
        return Agent(
            self._oap_role,
            self._oap_tool_manager,
            "",
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
        """Register python_exec tool when allow_code_execution is enabled on the class or any ancestor."""
        config = _collect_oap_config(self.__class__)
        if not config.get("allow_code_execution", False):
            return
        description = build_sandbox_description(config.get("imports"))
        self._oap_tool_manager.register_tool(
            Tool(
                name="python_exec",
                description=description,
                func=self._python_exec,
            )
        )

    def _register_media_tool(self) -> None:
        """Register read_media tool when allow_media_access is enabled on the class or any ancestor."""
        config = _collect_oap_config(self.__class__)
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
        config = _collect_oap_config(self.__class__)
        sandbox_globals = create_sandbox_globals(self, config)

        try:
            exec(code, sandbox_globals)
            return "OK"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool(name="produce_output", description="Produce the desired output and signal your final answer. Pass the result as a JSON string describing the output data.")
    def _produce_output(self, data: str, runner: "Runner | None" = None) -> str:
        """Protected tool: signals the agent has produced its final answer.

        Parses the JSON string and validates it against the current output
        schema. If validation fails, returns an error message the agent can
        use to correct its tool call.

        Args:
            data: The result as a JSON string.
            runner: The runner (injected by the execution environment).

        Returns:
            "OK" on success, or an error message the agent can fix.
        """
        if runner is None:
            _logger.debug("_produce_output: runner is None")
            return "Error: runner not available."
        try:
            parsed = parse_data(data, self._oap_current_output_schema)
        except ValueError as e:
            desc = get_schema_description(self._oap_current_output_schema)
            if desc is None:
                hint = f"Expected a value of type {self._oap_current_output_schema.__name__}."
            else:
                hint = f"Expected data matching schema: {desc[0]}."
                if desc[1]:
                    hint += f"\nSchema description: {desc[1]}."
            return f"{e}\n{hint}"

        try:
            if runner.state.get("_oap_produced_data"):
                runner.state.delete("_oap_produced_data")
            runner.state.create("_oap_produced_data", data)
            _logger.debug(
                "_produce_output: wrote to runner %s, _oap_produced_data=%s",
                runner.session_uuid, parsed,
            )
        except ValueError as e:
            return f"Error: {e}"
        return None

    async def _read_media(self, src: str, runner: "Runner | None" = None) -> str:
        """Tool: load a media file and queue it back to the agent's runner.

        Reads the file from disk or fetches from a URL, encodes it as a
        ContentPart (image/video/pdf), and queues a new user message into
        the runner's event queue. The runner loop will drain this message
        and add it to chat history before the next LLM call.

        Args:
            src: Local file path or HTTP(S) URL to the media file.
            runner: The runner (injected by the execution environment).

        Returns:
            Confirmation message with file info.
        """
        from peteos.conversation import ContentPart, Message

        if runner is None:
            return "Error: runner not available."

        from peteos.conversation.media import create_media_content_part_async
        media_part = await create_media_content_part_async(src, timeout=30.0)

        queued_msg = Message.create(
            role="user",
            content_parts=[
                ContentPart.create_text(f"Media loaded from {src}."),
                media_part,
            ],
        )
        await runner.queue_message(queued_msg)
        _logger.debug("_read_media: queued media from %s for runner %s", src, runner.session_uuid)
        return f"OK: media queued from {src}"

    async def _send_media(self, data: bytes, mime_type: str, runner: "Runner | None", *, text: str | None = None) -> None:
        """Send in-memory media bytes as a user message to the runner.

        Encodes the bytes as base64, creates a ContentPart, and queues it
        as a user message in the runner's event queue. Useful for passing
        media captured from cameras, memory, or other non-file sources.

        Args:
            data: Raw media bytes (e.g. JPEG/PNG file data).
            mime_type: MIME type of the media (e.g. "image/jpeg", "video/mp4").
            runner: The runner (injected by the execution environment).
            text: Optional text message to include alongside the media.
        """
        from peteos.conversation import ContentPart, Message

        if runner is None:
            return

        from peteos.conversation.media import create_media_content_part_async
        media_part = await create_media_content_part_async(data, mime_type=mime_type)

        msg_text = text or "Media sent."
        queued_msg = Message.create(
            role="user",
            content_parts=[
                ContentPart.create_text(msg_text),
                media_part,
            ],
        )
        await runner.queue_message(queued_msg)
        _logger.debug("_send_media: queued media (type=%s) for runner %s", mime_type, runner.session_uuid)

    @tool(name="produce_error", description="Signal that you could not produce the requested output. Pass an error message explaining why (e.g., missing required data or an invalid state).")
    def _produce_error(self, message: str, runner: "Runner | None" = None) -> str:
        """Protected tool: signals the agent could not fulfill the task.

        Writes the error message to the runner's AgenticState so that
        invoke_agent returns an Error object.

        Args:
            message: Human-readable error explanation.
            runner: The runner (injected by the execution environment).

        Returns:
            "OK" on success, or an error message if the runner is missing.
        """
        if runner is None:
            return "Error: runner not available."
        try:
            runner.state.create("_oap_error", message)
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

        Acquires the invocation lock, creates or reuses a Session and Runner,
        queues the prompt, waits for produce_output (via after_step hook),
        extracts structured output, then releases the lock.

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
        _logger.debug("invoke_agent[%s]: gatekeeper passed", self.__class__.__name__)

        # --- Acquire lock with timeout ---
        self.acquire(timeout)
        _logger.debug("invoke_agent[%s]: lock acquired", self.__class__.__name__)

        # --- Update output schema (serialized by lock) ---
        self._oap_current_output_schema = output_schema

        # --- Create or reuse runner (session + runner pair) ---
        session: Session | None = None
        runner: Runner | None = None
        if persistent_thread_id is not None:
            stored_uuid = self._oap_thread_store.get(persistent_thread_id)
            if stored_uuid is not None:
                _logger.debug("invoke_agent[%s]: reusing thread %s", self.__class__.__name__, persistent_thread_id)
                session = self._oap_agent.get_session(stored_uuid)
                if session is not None:
                    runner = Runner(self._oap_agent, session.uuid)
                    _logger.debug("invoke_agent[%s]: starting existing runner", self.__class__.__name__)
                    await runner.start()
        if session is None or runner is None:
            _logger.debug("invoke_agent[%s]: creating new session and runner", self.__class__.__name__)
            session = await self._oap_agent.create_session()
            runner = Runner(self._oap_agent, session.uuid)
            _logger.debug("invoke_agent[%s]: starting new runner", self.__class__.__name__)
            await runner.start()
            if persistent_thread_id is not None:
                self._oap_thread_store[persistent_thread_id] = session.uuid

        try:
            async def _on_step_done(r: Runner, status: ExecStatus) -> ExecStatus | None:
                produced = r.state.get("_oap_produced_data")
                errored = r.state.get("_oap_error")
                if errored is not None:
                    _logger.debug("_on_step_done: error found, returning FINISHED")
                    return ExecStatus.FINISHED
                elif produced is not None:
                    _logger.debug("_on_step_done: produced data found, returning FINISHED")
                    return ExecStatus.FINISHED
                elif status == ExecStatus.FINISHED:
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
                        f"If you have your final answer, call `produce_output` with your result in the following schema: {schema_desc} "
                    )
                    await r.queue_message(Message.create(
                        role="user",
                        content_parts=[ContentPart.create_text(reminder)],
                    ))
                return None

            runner.execution_environment.register_hook(
                "after_step", _on_step_done, runner
            )
            _logger.debug("invoke_agent[%s]: after_step hook registered", self.__class__.__name__)

            content: list[ContentPart] = [ContentPart.create_text(prompt)]
            if image is not None:
                from peteos.conversation.media import create_media_content_part_async
                image_part = await create_media_content_part_async(image, timeout=timeout or 30.0)
                content.append(image_part)

            _logger.debug("invoke_agent[%s]: queuing prompt message", self.__class__.__name__)
            await runner.queue_message(Message.create(
                role="user",
                content_parts=content,
            ))

            reminder_msg = f"PRODUCE YOUR FINAL OUTPUT USING THE `produce_output` OR `produce_error` TOOL!\n" \
                           f"MIND THE OUTPUT SCHEMA!"
            start_time = time.time()

            iteration = 0
            while True:
                iteration += 1
                _logger.debug("invoke_agent[%s]: waiting for idle (iter %d)", self.__class__.__name__, iteration)
                await runner.wait_for_idle(timeout=timeout)
                _logger.debug("invoke_agent[%s]: idle reached (iter %d), checking state", self.__class__.__name__, iteration)
                produced_data = runner.state.get("_oap_produced_data")
                if produced_data is not None:
                    _logger.debug("invoke_agent[%s]: produced_data found, returning", self.__class__.__name__)
                    try:
                        return parse_data(produced_data, self._oap_current_output_schema)
                    except ValueError as e:
                        _logger.warning("invoke_agent[%s]: produced data failed to parse: %s", self.__class__.__name__, e)
                        return Error(f"Agent produced data that failed schema validation: {e}")
                error_msg = runner.state.get("_oap_error")
                if error_msg is not None:
                    _logger.debug("invoke_agent[%s]: error found, returning", self.__class__.__name__)
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

                _logger.debug("invoke_agent[%s]: queuing reminder (iter %d)", self.__class__.__name__, iteration)
                await runner.queue_message(Message.create(
                    role="user",
                    content_parts=[ContentPart.create_text(reminder_msg)],
                ))
        finally:
            # Clear OAP state variables so the next invocation starts fresh
            if runner is not None:
                try:
                    produced = runner.state.get("_oap_produced_data")
                    if produced is not None:
                        runner.state.delete("_oap_produced_data")
                except KeyError:
                    pass
                try:
                    error = runner.state.get("_oap_error")
                    if error is not None:
                        runner.state.delete("_oap_error")
                except KeyError:
                    pass
            if persistent_thread_id is None:
                try:
                    await runner.stop()
                    await self._oap_agent.destroy_session(session.uuid)
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
        config = _collect_oap_config(self.__class__)
        if not config.get("invoke_sub_agents", False):
            return Error("Sub-agent invocation not enabled")

        # --- Forward to target's invoke_agent ---
        return await target.invoke_agent(
            prompt=prompt,
            output_schema=output_schema,
            timeout=timeout,
        )
