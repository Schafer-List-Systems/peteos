"""OAP invoke_agent() — main entry point composing with peteos infrastructure."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any

from peteos.agent import Agent
from peteos.chatbot import ContentPart, Message
from peteos.role import Role
from peteos.session import invoke_agent as _peteos_invoke_agent
from peteos.toolmanager import Tool
from peteos.toolmanager import ToolManager

from peteos.oap._schema import (
    extract_schema_info,
    format_schema_prompt,
    parse_output,
)
from peteos.oap._thread_store import ThreadStore
from peteos.oap.sandbox import create_sandbox_globals

if TYPE_CHECKING:
    from peteos.oap.base import AgenticObjectBase

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 180.0


async def invoke(
    oap_object: "AgenticObjectBase",
    prompt: str = "",
    output_schema: type | None = None,
    thread_id: str | None = None,
    agent: Agent | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Invoke an OAP agent and return structured output.

    Returns exactly one of:
    - {"result": <value>, "success": True, "thread_id": str}
    - {"error": Error, "success": False, "thread_id": str}
    - Raises Exception for API-level failures

    Args:
        oap_object: The AgenticObjectBase instance to invoke.
        prompt: The user prompt / task for the agent.
        output_schema: Optional type for structured output parsing.
        thread_id: None for non-persistent, or a string for persistent threads.
        agent: Optional Agent instance. Falls back to oap_object._oap_agent.
        timeout: Maximum seconds to wait (default: 60).

    Raises:
        ValueError: If no agent is available.
        asyncio.TimeoutError: If timeout expires.
        RuntimeError: For other API failures.
    """
    from peteos.oap.error import Error as OapError

    # --- Agent resolution ---
    resolved_agent = agent or oap_object.agent
    if resolved_agent is None:
        raise ValueError(
            "No Agent available: pass agent= or set oap_object.agent"
        )

    # --- Discover tools ---
    oap_tool_manager = _discover_tools(oap_object)

    # --- Generate system prompt ---
    system_prompt = _generate_system_prompt(oap_object, output_schema)

    # --- Create dynamic Role ---
    class_name = oap_object.__class__.__name__
    dynamic_role = _create_dynamic_role(class_name, system_prompt, oap_tool_manager)

    # --- Register dynamic role with the agent's RoleManager ---
    role_manager = resolved_agent._role_manager
    if role_manager.get_role(dynamic_role.name) is None:
        role_manager.register_role(dynamic_role)

    # --- Thread resolution ---
    thread_created = False
    persistent_session_uuid: Any = None

    if thread_id is not None:
        existing_session = ThreadStore.get_session(thread_id)
        if existing_session is None:
            existing_session = await resolved_agent.create_session(dynamic_role.name)
            ThreadStore.set_session(thread_id, existing_session, resolved_agent)
            persistent_session_uuid = existing_session.uuid
    else:
        existing_session = None

    try:
        # --- Invoke via peteos infrastructure ---
        result = await _peteos_invoke_agent(
            role_name=dynamic_role.name,
            prompt=prompt,
            agent=resolved_agent,
            existing_session=existing_session,
            keep_session=(thread_id is not None),
            timeout=timeout,
        )

        answer = result.get("answer", "")

        # --- Post-process: parse structured output ---
        if output_schema is not None:
            parsed = parse_output(answer, output_schema)
            if isinstance(parsed, OapError):
                return {"result": parsed, "success": False, "thread_id": thread_id or ""}

        # --- Post-process: detect task failure indicators ---
        if _detect_task_failure(answer):
            return {
                "result": OapError(answer[:500]),
                "success": False,
                "thread_id": thread_id or "",
            }

        # --- Return result ---
        return {
            "result": parse_output(answer, output_schema) if output_schema else answer,
            "success": True,
            "thread_id": thread_id or "",
        }

    except Exception:
        raise
    finally:
        # --- Cleanup persistent thread ---
        if thread_id is not None and persistent_session_uuid is not None:
            session = resolved_agent.get_session(persistent_session_uuid)
            if session is not None and session.is_running():
                try:
                    await session.stop()
                except Exception:
                    pass
            ThreadStore.destroy(thread_id)


def _discover_tools(oap_object: "AgenticObjectBase") -> ToolManager:
    """Discover @tool-decorated methods from the object class.

    Returns a ToolManager with discovered tools registered.
    """
    tm = ToolManager()
    cls = oap_object.__class__

    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if not callable(attr):
            continue
        if not hasattr(attr, "_tool_name"):
            continue

        tool_name = attr._tool_name  # type: ignore[attr-defined]
        tool_desc = attr._tool_description or ""  # type: ignore[attr-defined]

        # Bind the method to this instance so self is supplied automatically
        bound_func = getattr(oap_object, attr_name)

        # Extract parameter info from the method signature
        parameters = _extract_method_params(attr)

        tool = Tool(
            name=tool_name,
            description=tool_desc,
            func=bound_func,
            parameters=parameters,
        )
        tm.register_tool(tool=tool)

    return tm


def _extract_method_params(method: Any) -> dict:
    """Extract parameter schema from a method for the LLM."""
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return {}

    # Skip 'self' and 'session' parameters
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


def _extract_data_state(oap_object: "AgenticObjectBase") -> str:
    """Extract a snapshot of the object's data for the LLM's context.

    Iterates over all @tool methods returning a non-callable value
    (getter-like methods) and formats them as key-value lines.
    """
    cls = oap_object.__class__
    lines: list[str] = []
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if not callable(attr) or not hasattr(attr, "_tool_name"):
            continue
        # Skip methods that take arguments (they're setters, not getters)
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            continue
        params = [p for p in sig.parameters.keys() if p != "self"]
        if not params:
            try:
                value = attr(oap_object)
            except Exception:
                value = "<error reading value>"
            lines.append(f"  {attr._tool_name}: {value!r}")
    return "\n".join(lines)


def _build_tool_descriptions(oap_object: "AgenticObjectBase") -> list[str]:
    """Build a list of tool descriptions from decorated methods."""
    cls = oap_object.__class__
    result: list[str] = []
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if not callable(attr) or not hasattr(attr, "_tool_name"):
            continue
        desc = getattr(attr, "_tool_description", "") or ""
        tool_name = attr._tool_name
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            result.append(f"  - `{tool_name}`: {desc}")
            continue
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            ptype = ""
            if param.annotation != inspect.Parameter.empty:
                ptype = str(param.annotation)
            params.append(f"  - `{pname}` ({ptype})")
        param_str = "\n".join(params) if params else "  (no parameters)"
        desc_str = f": {desc}" if desc else ""
        result.append(f"  - `{tool_name}`{desc_str}\n{param_str}")
    return result


def _generate_system_prompt(
    oap_object: "AgenticObjectBase",
    output_schema: type | None,
) -> str:
    """Generate a dynamic system prompt describing the object's interface."""
    lines: list[str] = []

    # Class description
    docstring = oap_object.__class__.__doc__
    if docstring:
        lines.append(docstring.strip())
        lines.append("")

    # Current data state
    config = getattr(oap_object.__class__, "_oap_config", {})
    state = _extract_data_state(oap_object)
    if state:
        lines.append("Current data state:")
        lines.append(state)
        lines.append("")

    # Capabilities
    capabilities = []

    if config.get("allow_code_execution", False):
        capabilities.append(
            "You have access to code execution. Use the `_run_code` tool to "
            "execute Python code. The sandbox provides `self` as the only "
            "object reference and no built-in functions."
        )
    if config.get("invoke_sub_agents", False):
        capabilities.append(
            "You can invoke sub-agents on child AgenticObjectBase instances "
            "that have `invoke_sub_agents=True` via their `invoke()` method."
        )

    if capabilities:
        lines.append("Capabilities:")
        for cap in capabilities:
            lines.append(f"  - {cap}")
        lines.append("")

    # Tool descriptions
    tool_lines = _build_tool_descriptions(oap_object)
    if tool_lines:
        lines.append("Available tools:")
        lines.extend(tool_lines)
        lines.append("")

    # Output schema instructions
    schema_info = extract_schema_info(output_schema)
    schema_prompt = format_schema_prompt(schema_info)
    if schema_prompt:
        lines.append(schema_prompt)
        lines.append("")

    lines.append("Terminate when you have produced the requested output.")

    return "\n".join(lines)


def _create_dynamic_role(
    name: str,
    system_prompt: str,
    tool_manager: ToolManager,
) -> Role:
    """Create a peteos Role for OAP invocations.

    The dynamic Role gets auto_approve_tools set to all discovered tool names
    so the Agent's approval hooks don't block execution.
    """
    all_tools = tool_manager.get_tool_list()
    tool_names = [t.name for t in all_tools]

    # Also include the _run_code tool if code execution is enabled
    if "_run_code" not in tool_names:
        tool_names.append("_run_code")

    return Role(
        name=f"oap_{name}",
        description=f"OAP agent for {name}",
        system_prompt=system_prompt,
        auto_approve_tools=tool_names,
        tool_filter=tool_names,
        required_tools=tool_names,
    )


def _detect_task_failure(answer: str) -> bool:
    """Heuristic: detect if the LLM's response indicates task failure.

    Checks for patterns like "I cannot", "Error:", "Unable to" in the response.
    """
    lower = answer.lower().strip()
    failure_patterns = [
        "error:",
        "i cannot",
        "i'm unable",
        "unable to",
        "i don't have",
        "no such",
        "not found",
    ]
    return any(p in lower for p in failure_patterns)
