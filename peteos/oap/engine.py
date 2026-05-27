"""OAP invoke_agent() — thin wrapper around peteos infrastructure."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Any

from peteos.agent import Agent
from peteos.chatbot import ContentPart, Message
from peteos.executionenvironment import ExecStatus
from peteos.role import Role
from peteos.toolmanager import Tool
from peteos.toolmanager import ToolManager

from peteos.oap._schema import extract_schema_info
from peteos.oap._thread_store import ThreadStore

if TYPE_CHECKING:
    from peteos.oap.base import AgenticObjectBase

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180.0
_MAX_ITERATIONS = 10


async def invoke(
    oap_object: "AgenticObjectBase",
    prompt: str = "",
    output_schema: type | None = None,
    thread_id: str | None = None,
    agent: Agent | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Invoke an OAP agent and return structured output.

    If output_schema is provided, a synthetic produce_output() tool is
    registered and the agent loops until it calls produce_output()
    (with a valid result) or reports an error.
    """
    from peteos.oap.error import Error as OapError

    # --- Agent resolution ---
    resolved_agent = agent or oap_object.agent
    if resolved_agent is None:
        raise ValueError(
            "No Agent available: pass agent= or set oap_object.agent"
        )

    # --- Discover tools and bind to instance ---
    bound_tools = _discover_bound_tools(oap_object)
    tool_names = [t.name for t in bound_tools]

    # --- Output schema handling ---
    schema_info = extract_schema_info(output_schema)
    has_output_schema = schema_info is not None
    system_prompt = ""
    if oap_object.__class__.__doc__:
        system_prompt = oap_object.__class__.__doc__.strip()

    # --- Synthetic produce_output() tool ---
    produce_callback: list[dict[str, Any] | None] = [None]
    produce_func: Any = None

    if has_output_schema:
        produce_func = _build_produce_func(produce_callback, schema_info)
        tool_names.append("produce_output")

        system_prompt += (
            "\n\nWhen you are ready to report your final result, "
            "call produce_output() with the structured data matching "
            "the output schema. If you cannot produce the result, "
            "call produce_output() with an error message instead."
            "\n\nIMPORTANT: Call produce_output() to finish. Do not produce text output without calling produce_output()."
        )

    # --- Create a minimal Role ---
    class_name = oap_object.__class__.__name__
    dynamic_role = Role(
        name=f"oap_{class_name}",
        description=f"OAP agent for {class_name}",
        system_prompt=system_prompt,
        auto_approve_tools=tool_names,
        tool_filter=tool_names,
        required_tools=tool_names,
    )

    # --- Register with RoleManager ---
    role_manager = resolved_agent._role_manager
    if role_manager.get_role(dynamic_role.name) is None:
        role_manager.register_role(dynamic_role)

    # --- Register tools on agent's tool_manager ---
    for tool in bound_tools:
        resolved_agent._tool_manager.register_tool(tool=tool)

    # --- Session resolution ---
    persistent_session_uuid: Any = None
    temp_session_uuid: Any = None
    session_to_use: Any = None

    if thread_id is not None:
        existing = ThreadStore.get_session(thread_id)
        if existing is None:
            session_to_use = await resolved_agent.create_session(dynamic_role.name)
            ThreadStore.set_session(thread_id, session_to_use, resolved_agent)
            persistent_session_uuid = session_to_use.uuid
    else:
        session_to_use = await resolved_agent.create_session(dynamic_role.name)
        temp_session_uuid = session_to_use.uuid

    try:
        # --- Run the loop ---
        return await _run_oap_loop(
            resolved_agent,
            produce_func,
            produce_callback,
            prompt,
            thread_id,
            timeout,
            has_output_schema,
            session_to_use,
            temp_session_uuid,
        )

    except Exception:
        raise
    finally:
        # --- Cleanup ---
        if thread_id is not None and persistent_session_uuid is not None:
            sess = resolved_agent.get_session(persistent_session_uuid)
            if sess is not None and sess.is_running():
                try:
                    await sess.stop()
                except Exception:
                    pass
            ThreadStore.destroy(thread_id)
        elif temp_session_uuid is not None:
            sess = resolved_agent.get_session(temp_session_uuid)
            if sess is not None and sess.is_running():
                try:
                    await sess.stop()
                except Exception:
                    pass


def _build_produce_func(
    callback: list[dict[str, Any] | None],
    schema_info: dict[str, Any],
) -> Any:
    """Build and return a produce_output function."""
    properties = schema_info.get("properties", {})
    param_schemas: dict[str, dict[str, Any]] = {}
    for field_name, field_info in properties.items():
        param_schemas[field_name] = {
            "type": field_info.get("type", "string"),
            "required": field_name in schema_info.get("required", []),
            "description": field_info.get("description", ""),
        }

    def _produce_output(**kwargs: Any) -> None:
        callback[0] = kwargs

    _produce_output._oap_params = param_schemas  # type: ignore[attr-defined]
    return _produce_output


def _discover_bound_tools(oap_object: "AgenticObjectBase") -> list[Tool]:
    """Discover @tool-decorated methods and bind them to the object instance."""
    tools: list[Tool] = []
    cls = oap_object.__class__

    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if not callable(attr) or not hasattr(attr, "_tool_name"):
            continue

        tool_name = attr._tool_name  # type: ignore[attr-defined]
        tool_desc = attr._tool_description or ""  # type: ignore[attr-defined]
        bound_func = getattr(oap_object, attr_name)
        parameters = _extract_method_params(attr)

        tools.append(Tool(
            name=tool_name,
            description=tool_desc,
            func=bound_func,
            parameters=parameters,
        ))

    return tools


async def _run_oap_loop(
    agent: Agent,
    produce_func: Any,
    produce_callback: list[dict[str, Any] | None],
    prompt: str,
    thread_id: str | None,
    timeout: float,
    has_output_schema: bool,
    session_to_use: Any,
    temp_session_uuid: Any,
) -> dict[str, Any]:
    """Run OAP sessions until produce_output() is called."""
    from peteos.oap.error import Error as OapError

    # --- Register produce_output on the agent's tool_manager ---
    if produce_func is not None:
        agent._tool_manager.register_tool(
            Tool(
                name="produce_output",
                description="Report your final result",
                func=produce_func,
                parameters=getattr(produce_func, "_oap_params", {}),
            )
        )

    for iteration in range(_MAX_ITERATIONS):
        env = session_to_use.execution_environment
        done: asyncio.Event = asyncio.Event()

        def on_finished(sess_obj: Any, status: ExecStatus) -> None:
            if status == ExecStatus.FINISHED:
                done.set()

        env.register_hook("after_step", on_finished, None)

        try:
            if iteration == 0:
                user_text = prompt
            else:
                user_text = (
                    "You must call produce_output() to report your final "
                    "result or report an error. Do not produce text output."
                )

            await session_to_use.queue_message(
                Message(
                    role="user",
                    content=[ContentPart(part_type="text", text=user_text)],
                )
            )

            try:
                await asyncio.wait_for(done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

            # --- Check produce_output ---
            if produce_callback[0] is not None:
                kwargs = produce_callback[0]
                error_msg = kwargs.pop("message", None)
                if error_msg:
                    return {
                        "result": OapError(error_msg),
                        "success": False,
                        "thread_id": thread_id or "",
                    }
                return {
                    "result": kwargs,
                    "success": True,
                    "thread_id": thread_id or "",
                }

            # --- Non-schema mode: return answer ---
            if not has_output_schema:
                answer = _extract_last_assistant_text(session_to_use.chat_history)
                return {
                    "result": answer,
                    "success": True,
                    "thread_id": thread_id or "",
                }

            # --- Schema mode, not done yet: use rolling_window_discard ---
            session_to_use.chat_history.rolling_window_discard(
                trigger_threshold=100000,  # 100k tokens
            )

        finally:
            env.deregister_hook("after_step", on_finished)

    # --- Max iterations reached ---
    return {
        "result": OapError(
            "Agent could not produce structured output within "
            f"{_MAX_ITERATIONS} iterations."
        ),
        "success": False,
        "thread_id": thread_id or "",
    }


def _extract_last_assistant_text(chat_history: Any) -> str:
    """Extract the last assistant message text from chat history."""
    for msg in reversed(chat_history.messages):
        if msg.get_role() == "assistant":
            texts = [
                part.text for part in msg.content
                if part.type == "text" and part.text
            ]
            return " ".join(texts).replace("  ", " ")
    return ""


def _extract_method_params(method: Any) -> dict:
    """Extract parameter schema from a method for the LLM."""
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return {}

    parameters = {}
    for name, param in sig.parameters.items():
        if name in ("self", "session"):
            continue

        info: dict[str, Any] = {}

        if param.annotation != inspect.Parameter.empty:
            ann = param.annotation
            info["type"] = getattr(ann, "__name__", str(ann))
        else:
            info["type"] = "any"

        if param.default == inspect.Parameter.empty:
            info["required"] = True
        else:
            info["required"] = False
            info["default"] = param.default

        parameters[name] = info

    return parameters
