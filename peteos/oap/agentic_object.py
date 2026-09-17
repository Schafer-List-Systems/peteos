"""AgenticObject - base class for all OAP objects."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from dataclasses import is_dataclass
from typing import Any, Callable

from peteos.conversation import ContentPart, Message
from peteos.conversation.session import Session
from peteos.engine import ExecStatus, Runner
from peteos.persona.role import Role
from peteos.persona.toolmanager import Tool, ToolManager
from peteos.utils import get_logger
from peteos.oap.error import Error
from peteos.oap import prompts
from peteos.sandbox import SandboxBuilder

from peteos.oap.decorators import tool
from peteos.oap.agentic_registry import AgenticObjectRegistry

_logger = get_logger(__name__)

_AGENT_BASE_DIR = os.environ.get("PETEOS_AGENT_BASE_DIR", "/tmp/peteos")


def _collect_oap_config(cls: type) -> dict[str, Any]:
    """Collect and merge OAP config from all classes in the MRO that directly inherit from AgenticObject.

    Mirrors the reverse-MRO iteration pattern used in _build_system_prompt. Collects the union of
    all imports, merges import_aliases, and ORs all boolean flags across the
    diamond hierarchy so that a class D(B, C) where both B and C define
    @agentic_object with different imports gets all of them combined.
    """
    from peteos.oap.agentic_registry import _is_oap_object

    imports: set[object] = set()
    import_aliases: dict[str, str] = {}
    allow_code_execution = False
    allow_media_access = False
    invoke_sub_agents = False
    define_functions = False
    for c in cls.__mro__:
        if c in (AgenticObject, object):
            continue
        if not _is_oap_object(c):
            continue
        cfg = getattr(c, "_oap_config", None)
        if cfg:
            allow_code_execution |= cfg.get("allow_code_execution", False)
            allow_media_access |= cfg.get("allow_media_access", False)
            invoke_sub_agents |= cfg.get("invoke_sub_agents", False)
            define_functions |= cfg.get("define_functions", False)
            imports.update(cfg.get("imports", []))
            import_aliases.update(cfg.get("import_aliases", {}))
    return {
        "allow_code_execution": allow_code_execution,
        "allow_media_access": allow_media_access,
        "invoke_sub_agents": invoke_sub_agents,
        "define_functions": define_functions,
        "imports": list(imports),
        "import_aliases": import_aliases,
    }


from peteos.utils._schema import _format_docs, get_schema_description, relaxed_parse_data


from peteos.persona.agent import Agent


class AgenticObject:
    """Base class for all Object-Agentic Programming objects.

    All subclasses are auto-registered in AgenticObjectRegistry.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        AgenticObjectRegistry.register(cls)

    def __init__(self) -> None:
        super().__init__()
        self._oap_role: Role = AgenticObjectRegistry.create_role(self.__class__.__name__)
        self._oap_lock: threading.Lock = threading.Lock()
        self._oap_tool_manager: ToolManager = ToolManager()
        self._oap_current_output_schema: type | None = None
        self._oap_system_prompt_hooks: dict[str, Callable[[], str]] = {}
        self._oap_thread_store: dict[str, str] = {}
        self._oap_auto_approve_tools: list[str] = []
        self._register_tools()
        self._register_output_schema_hook()
        self._register_sandbox_hook()
        self._oap_sandbox_builder: SandboxBuilder = self._init_sandbox_builder(_collect_oap_config(self.__class__))
        self._register_sandbox_tool()
        self._register_media_tool()
        self._oap_agent: Agent = self._create_agent()
        tool_names = [t.name for t in self._oap_tool_manager.get_tool_list()]
        _logger.debug("Registered tools for %s: %s", self.__class__.__name__, tool_names)

    def _create_agent(self) -> Agent:
        """Create an Agent wired to this object's role and tool_manager."""
        # Auto-approve all registered tools so they pass Runner.add_tool_call()
        for t in self._oap_tool_manager.get_tool_list():
            self._oap_auto_approve_tools.append(t.name)
        return Agent(
            self._oap_role,
            self._oap_tool_manager,
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
        self._oap_system_prompt_hooks["output_schema"] = self._output_schema_hook

    def _output_schema_hook(self) -> str:
        """System prompt hook: returns formatted output schema description."""
        output_schema = self._oap_current_output_schema
        desc = get_schema_description(output_schema)
        json_schema, doc_entries = desc
        return _format_docs(doc_entries)

    def _register_sandbox_hook(self) -> None:
        """Register the python_exec system prompt hook when code execution is enabled."""
        config = _collect_oap_config(self.__class__)
        if not config.get("allow_code_execution", False):
            return
        self._oap_system_prompt_hooks["python_exec"] = self._python_exec_system_prompt_hook

    def _python_exec_system_prompt_hook(self) -> str:
        """System prompt hook: instructs the agent on using the python_exec tool."""
        imports = _collect_oap_config(self.__class__).get("imports")
        imports_str = ""
        if imports:
            mods_list = ", ".join(
                m.__name__ if hasattr(m, "__name__") else str(m) for m in imports
            )
            imports_str = f"\nAvailable modules: {mods_list}."
        return prompts.PYTHON_EXEC_PROMPT.format(modules_section=imports_str)

    def _register_sandbox_tool(self) -> None:
        """Register python_exec tool when allow_code_execution is enabled on the class or any ancestor."""
        config = _collect_oap_config(self.__class__)
        if not config.get("allow_code_execution", False):
            return

        description = self._oap_sandbox_builder.build_sandbox_description()
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

    def _python_exec(self, function: str, runner: Runner) -> str:
        """Protected tool: executes sandboxed Python code."""
        return self._call_sandboxed(function, runner=runner)

    def _gather_sandbox_members(self) -> dict[str, Callable]:
        """Collect all @tool/@sandbox decorated member functions of this object.

        Walks the MRO and the instance __dict__ to find methods decorated
        with @tool or @sandbox, returning them as a dict mapping sandbox
        name to bound method. Each returned callable is already bound —
        self is stripped from the signature so the caller receives a
        ready-to-call function taking only the remaining parameters.
        The instance __dict__ takes precedence over class-level methods,
        shadowing for that particular instance.
        """
        members: dict[str, Callable] = {}

        def _register(name: str, method: Callable) -> None:
            sandbox_name = getattr(method, "_sandbox_name", None) or getattr(method, "_tool_name", None)
            if sandbox_name is None:
                return
            members[sandbox_name] = getattr(self, name)

        for cls in self.__class__.__mro__:
            for method_name, method in cls.__dict__.items():
                if callable(method):
                    _register(method_name, method)
        for method_name, method in self.__dict__.items():
            if callable(method):
                _register(method_name, method)
        return members

    def _init_sandbox_builder(self, config: dict[str, Any]) -> SandboxBuilder:
        """Create and initialize a SandboxBuilder with builtins and imports.

        Args:
            config: MRO-merged OAP config dict.

        Returns:
            A configured SandboxBuilder ready for further configuration.
        """
        builder = SandboxBuilder("instance")
        builder.add_safe_builtins()

        # Imports from config (MRO-merged).
        builder.add_imports(config.get("imports", []), config.get("import_aliases"))

        # All @tool/@sandbox decorated member functions from MRO + instance __dict__.
        members = self._gather_sandbox_members()
        builder.add_proxies(members)

        return builder

    def _create_sandbox_builder(
        self,
        config: dict[str, Any],
        runner: "Runner | None",
    ) -> SandboxBuilder:
        """Create and configure a SandboxBuilder for session sandbox code.

        Args:
            config: MRO-merged OAP config dict.
            runner: The runner (injected by the execution environment).

        Returns:
            A configured SandboxBuilder ready for :meth:`build`.
        """
        builder = SandboxBuilder("session", base=self._oap_sandbox_builder)

        # add proxies for produce_output, produce_error and invoke tools but with runner registered already
        def _produce_output(answer: Any) -> str | None:
            _logger.debug("build_session_sandbox: _produce_output wrapper called, runner=%s", runner)
            return self._produce_output(answer, runner=runner)

        def _produce_error(message: Any) -> str:
            return self._produce_error(message, runner=runner)  # type: ignore[arg-type]

        builder.add_proxy("produce_output", _produce_output)
        builder.add_proxy("produce_error", _produce_error)

        if config.get("invoke_sub_agents", False):
            parent_ptid = runner.state.get("_persistent_thread_id") if runner else None
            parent_hooks = runner.session.invocation_hooks if runner and runner.session else {}

            def _invoke(
                target,
                prompt,
                output_schema=None,
                persistent=False,
                timeout=None,
            ):
                ptid = parent_ptid if persistent else None

                def _run():
                    _loop = asyncio.new_event_loop()
                    try:
                        return _loop.run_until_complete(
                            target.invoke_agent(
                                prompt=prompt,
                                output_schema=output_schema,
                                timeout=timeout,
                                persistent_thread_id=ptid,
                                hooks=parent_hooks,
                            )
                        )
                    finally:
                        _loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(_run).result()

            builder.add_proxy("invoke", _invoke)

        return builder

    def _call_sandboxed(
        self,
        code: str,
        runner: Runner,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Compile code in an exec-level sandbox and invoke the matching function.

        Creates a SandboxBuilder with the runner's session-level sandbox as base,
        compiles the code, finds the matching member function by args/kwargs,
        calls it, and returns the result.
        """
        session_builder = runner.sandbox_builder
        if session_builder is None:
            raise ValueError(
                f"Session sandbox builder not set on runner — "
                "ensure _start_session was called before sandboxed execution"
            )
        builder = SandboxBuilder("exec", base=session_builder)
        entries = builder.add_source_code(code)

        # Find matching function by arg/kwargs count
        n = len(args) + len(kwargs)
        matching: list[str] = []
        for name, params in entries:
            param_count = len(params)
            if param_count == n:
                matching.append(name)

        if len(matching) != 1:
            raise ValueError(
                f"expected exactly one function, found {len(matching)}: {matching}"
            )

        sandbox = builder.get_sandbox()
        func = getattr(sandbox, matching[0])
        return func(*args, **kwargs)

    @tool(
        name="produce_output",
        description=(
            "Produce the desired output and signal your final answer. "
            "Pass the answer matching the output schema."
        )
    )
    def _produce_output(self, answer: Any, runner: Runner) -> str | None:
        """Protected tool: signals the agent has produced its final answer.

        Validates the answer against the current output schema. If validation
        fails, returns an error message the agent can use to correct its
        tool call.

        Args:
            answer: The result as a structured value (dict, list, or scalar).
            runner: The runner (injected by the execution environment).

        Returns:
            "OK" on success, or an error message the agent can fix.
        """
        if runner is None:
            _logger.debug("_produce_output: runner is None")
            return "Error: runner not available."
        try:
            parsed = relaxed_parse_data(answer, self._oap_current_output_schema)
        except ValueError as e:
            attempts = runner.state.get("_oap_output_attempts") + 1
            max_attempts = self._oap_role.max_output_attempts
            if attempts >= max_attempts:
                _logger.error(
                    "invoke_agent[%s]: max output attempts hit after %d turns without calling produce_output or produce_error (limit=%d)",
                    self.__class__.__name__, attempts, max_attempts,
                )
                runner.set_critical_error(RuntimeError(
                    f"Agent exceeded max output attempts ({max_attempts}) by finishing steps without successfully calling `produce_output` or `produce_error`"
                ))
                return None
            runner.state.force_set("_oap_output_attempts", attempts)

            output_schema = self._oap_current_output_schema

            desc = get_schema_description(output_schema)
            json_schema, doc_entries = desc
            hint = f"Expected answer matching schema format: {json_schema}."
            docstring = _format_docs(doc_entries)
            if docstring:
                hint += f"\nSchema description:\n{docstring}"
            return f"{e}\n{hint}"

        try:
            if runner.state.get("_oap_produced_data"):
                runner.state.delete("_oap_produced_data")
            runner.state.create("_oap_produced_data", answer)
            _logger.debug(
                "_produce_output: wrote to runner %s, _oap_produced_data=%s",
                runner.session_uuid, parsed,
            )
        except ValueError as e:
            return f"Error: {e}"
        return None

    async def _read_media(self, src: str, runner: Runner) -> str:
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

    async def _send_media(self, data: bytes, mime_type: str, runner: Runner, *, text: str | None = None) -> None:
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
    def _produce_error(self, message: Any, runner: Runner = None) -> str | None:
        """Protected tool: signals the agent could not fulfill the task.

        Writes the error message to the runner's SessionState so that
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

    @property
    def agent(self) -> Agent | None:
        """The Agent instance used for LLM invocations."""
        return self._oap_agent

    @property
    def role(self) -> Role:
        """The Role used for this object's invocations."""
        return self._oap_role

    async def _start_session(self, session: Session) -> Runner:
        runner = Runner(self._oap_agent, session.uuid)
        runner._execution_environment.auto_approve_tools = list(self._oap_auto_approve_tools)
        runner.sandbox_builder = self._create_sandbox_builder(
            _collect_oap_config(self.__class__), runner
        )
        runner.sandbox = runner.sandbox_builder.get_sandbox(freeze_namespaces=False)
        await runner.start()
        session.is_active = True
        return runner

    async def invoke_agent(
        self,
        prompt: str,
        output_schema: type | None = None,
        persistent_thread_id: str | None = None,
        timeout: float | None = None,
        image: str | None = None,
        hooks: dict[str, list[Callable]] | None = None,
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
            hooks: Optional dictionary of hook names to lists of callables.
                These hooks are stored on the session for the duration of the
                invocation and forwarded recursively to sub-agents.

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
                    if session.is_active:
                        return Error(
                            f"Session {stored_uuid} is already active; "
                            "recursive invoke_agent on the same persistent thread is not allowed"
                        )
                    runner = await self._start_session(session)
        if session is None or runner is None:
            _logger.debug("invoke_agent[%s]: creating new session and runner", self.__class__.__name__)
            session = await self._oap_agent.create_session()
            system_prompt_msg = session.active_context.system_prompt_message
            for name, callback in self._oap_system_prompt_hooks.items():
                session.register_hook(system_prompt_msg, name, callback)
            runner = await self._start_session(session)
            if persistent_thread_id is not None:
                self._oap_thread_store[persistent_thread_id] = session.uuid

        if persistent_thread_id is not None:
            try:
                runner.state.create("_persistent_thread_id", persistent_thread_id)
            except ValueError:
                pass  # key may already exist from a prior invoke_agent call on the same persistent session

        # Store invocation hooks on the session (clear first, then merge)
        session._invocation_hooks = {}
        if hooks is not None:
            session._invocation_hooks.update(hooks)
        _logger.debug("invoke_agent[%s]: stored %d hooks on session %s", self.__class__.__name__, len(session._invocation_hooks), session.uuid)

        # Fire on_invoke hooks — first non-None string prevents invocation
        _invocation_prevented: str | None = None
        if hooks:
            ctx = {
                "role": self._oap_role.name,
                "prompt": prompt,
                "session": session,
            }
            for hook in hooks.get("on_invoke", []):
                result = hook(ctx)
                if result is not None:
                    _invocation_prevented = result
                    _logger.debug(
                        "invoke_agent[%s]: invocation prevented by hook: %s",
                        self.__class__.__name__, result,
                    )
                    break
        final_result: Any = None
        try:
            if _invocation_prevented is not None:
                return Error(_invocation_prevented)

            async def _on_step_done(r: Runner, status: ExecStatus) -> ExecStatus | None:
                produced = r.state.get("_oap_produced_data")
                errored = r.state.get("_oap_error")

                # Attempt to use assistant's text as structured output for non-tool-call LLMs.
                # LLMs that cannot use tool calls may still provide valid JSON as text.
                # We try to parse that text and inject it via _produce_output so the normal
                # FINISHED check below can pick it up and return it to the caller.
                if errored is None and produced is None:
                    if self._oap_current_output_schema is not None and self._oap_current_output_schema is not Any:
                        for msg in reversed(r.session.active_context.messages):
                            if msg.role == "assistant":
                                text_parts = [cp.text for cp in msg.content if cp.type == "text"]
                                if text_parts:
                                    text = "".join(text_parts)
                                    _logger.debug(
                                        "_on_step_done: attempting _produce_output with LLM text for non-tool-call LLM"
                                    )
                                    self._produce_output(text, runner=r)
                                    break

                if errored is not None:
                    _logger.debug("_on_step_done: error found, returning FINISHED")
                    return ExecStatus.FINISHED
                elif produced is not None:
                    _logger.debug("_on_step_done: produced answer found, returning FINISHED")
                    return ExecStatus.FINISHED
                elif status == ExecStatus.FINISHED:
                    attempts = r.state.get("_oap_output_attempts") + 1
                    r.state.force_set("_oap_output_attempts", attempts)
                    max_attempts = self._oap_role.max_output_attempts
                    if attempts >= max_attempts:
                        _logger.error(
                            "invoke_agent[%s]: max output attempts hit after %d reminder turns (limit=%d)",
                            self.__class__.__name__, attempts, max_attempts,
                        )
                        raise RuntimeError(
                            f"Agent exceeded max output attempts ({max_attempts}) by finishing steps without successfully calling `produce_output` or `produce_error`"
                        )

                    output_schema = self._oap_current_output_schema
                    desc = get_schema_description(output_schema)
                    json_schema, doc_entries = desc
                    docstring = _format_docs(doc_entries)

                    schema_desc = f"Provide your final answer with the `produce_output` tool and arguments matching the schema:\n{json_schema}"
                    if docstring:
                        schema_desc += f"\nSchema description:\n{docstring}"
                    reminder = (
                        f"It looks like you finished a step without calling `produce_output` or `produce_error`. "
                        f"If you have your final answer, call `produce_output` with arguments in the following schema: {schema_desc} "
                    )

                    await r.queue_message(Message.create(
                        role="user",
                        content_parts=[ContentPart.create_text(reminder)],
                    ))
                return None

            if output_schema is not None and output_schema is not Any:
                session._invocation_hooks.setdefault("after_step", []).append(
                    lambda status: _on_step_done(runner, status)
                )
                _logger.debug(
                    "invoke_agent[%s]: after_step hook registered (strict schema)",
                    self.__class__.__name__,
                )

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
            runner.state.force_set("_oap_output_attempts", 0)
            final_result: Any = None
            while True:
                iteration += 1
                _logger.debug("invoke_agent[%s]: waiting for idle (iter %d)", self.__class__.__name__, iteration)
                await runner.wait_for_idle(timeout=timeout)
                _logger.debug("invoke_agent[%s]: idle reached (iter %d), checking state", self.__class__.__name__, iteration)

                # Check for critical error from step() — runner is in a fatal state, stop looping
                runner.take_critical_error()

                produced_data = runner.state.get("_oap_produced_data")
                if produced_data is not None:
                    _logger.debug("invoke_agent[%s]: produced_data found", self.__class__.__name__)
                    try:
                        final_result = relaxed_parse_data(produced_data, self._oap_current_output_schema)
                        return final_result
                    except ValueError as e:
                        _logger.warning("invoke_agent[%s]: produced answer failed to parse: %s", self.__class__.__name__, e)
                        final_result = Error(f"Agent produced answer that failed schema validation: {e}")
                    return final_result

                error_msg = runner.state.get("_oap_error")
                if error_msg is not None:
                    _logger.debug("invoke_agent[%s]: error found", self.__class__.__name__)
                    final_result = Error(error_msg)
                    return final_result

                max_turns = self._oap_role.max_output_turns
                if iteration > max_turns:
                    _logger.error(
                        "invoke_agent max_output_turns hit for %s thread=%s after %d turns (limit=%d)",
                        self.__class__.__name__,
                        persistent_thread_id,
                        iteration,
                        max_turns,
                    )
                    final_result = Error(f"Agent exceeded max output turns ({max_turns}) without producing output or error")
                    return final_result

                elapsed = time.time() - start_time
                if timeout is not None and elapsed > timeout:
                    _logger.error(
                        "invoke_agent timeout hit for %s thread=%s after %.1fs",
                        self.__class__.__name__,
                        persistent_thread_id,
                        elapsed,
                    )
                    raise TimeoutError(f"Agent did not produce output within {timeout}s timeout")

                if output_schema is None or output_schema is Any:
                    for msg in reversed(runner.session.active_context.messages):
                        if msg.role == "assistant":
                            text_parts = [cp.text for cp in msg.content if cp.type == "text"]
                            if text_parts:
                                text = "".join(text_parts)
                                if output_schema is Any:
                                    try:
                                        import json
                                        final_result = json.loads(text)
                                        _logger.debug(
                                            "invoke_agent[%s]: Any schema, parsed JSON (iter %d)",
                                            self.__class__.__name__,
                                            iteration,
                                        )
                                    except json.JSONDecodeError:
                                        final_result = text
                                        _logger.debug(
                                            "invoke_agent[%s]: Any schema, returning raw text (iter %d)",
                                            self.__class__.__name__,
                                            iteration,
                                        )
                                else:
                                    final_result = text
                                    _logger.debug(
                                        "invoke_agent[%s]: %s schema, returning text (iter %d)",
                                        self.__class__.__name__,
                                        output_schema,
                                        iteration,
                                    )
                                return final_result
                    if output_schema is None:
                        _logger.debug(
                            "invoke_agent[%s]: no output schema, no text, returning None (iter %d)",
                            self.__class__.__name__,
                            iteration,
                        )
                        return None
                    _logger.debug(
                        "invoke_agent[%s]: Any schema, no assistant text, queuing reminder (iter %d)",
                        self.__class__.__name__,
                        iteration,
                    )
                    await runner.queue_message(Message.create(
                        role="user",
                        content_parts=[ContentPart.create_text(
                            "I am still waiting for your final answer. Please provide your output."
                        )],
                    ))
                    continue

                _logger.debug("invoke_agent[%s]: queuing reminder (iter %d)", self.__class__.__name__, iteration)
                await runner.queue_message(Message.create(
                    role="user",
                    content_parts=[ContentPart.create_text(reminder_msg)],
                ))
        finally:
            # Fire on_invoke_complete hooks before cleanup
            if hooks and session is not None and final_result is not None:
                ctx = {
                    "role": self._oap_role.name,
                    "prompt": prompt,
                    "session": session,
                    "result": final_result,
                }
                for hook in hooks.get("on_invoke_complete", []):
                    hook(ctx)
                    _logger.debug("invoke_agent[%s]: fired on_invoke_complete hook", self.__class__.__name__)

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
            # Always deactivate the session so the next invoke can reuse it
            if session is not None:
                session.is_active = False
                session._invocation_hooks.clear()
                if session._autosave:
                    session.save()
                _logger.debug("invoke_agent[%s]: cleared invocation hooks on session %s", self.__class__.__name__, session.uuid)
            try:
                await runner.stop()
            except Exception:
                pass
            if persistent_thread_id is None:
                try:
                    await self._oap_agent.destroy_session(session.uuid)
                except Exception:
                    pass
            self.release()
