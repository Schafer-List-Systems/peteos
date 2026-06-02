from __future__ import annotations

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

from ._types import EdgeState, TaskState


class Task(AgenticObjectBase):
    """Each Task is an OAP object that wraps a JSON node object."""

    def __init__(self, node_data: dict, process_id: str | None = None) -> None:
        super().__init__()
        self._node_data = node_data
        self._outgoing_edges: list[Edge] = []
        self._incoming_edges: list[Edge] = []
        self._process_id = process_id

    def get_outgoing_edges(self) -> list[Edge]:
        return self._outgoing_edges

    def get_incoming_edges(self) -> list[Edge]:
        return self._incoming_edges

    def get_successor_tasks(self) -> list[Task]:
        return [edge.get_to_task() for edge in self._outgoing_edges]

    def get_predecessor_tasks(self) -> list[Task]:
        return [edge.get_from_task() for edge in self._incoming_edges]

    @property
    def node_data(self) -> dict:
        return self._node_data

    @property
    def task_id(self) -> str:
        return self._node_data["id"]

    _COLORS: dict[TaskState, str] = {
        TaskState.SCHEDULED: "#808080",
        TaskState.ACTIVE: "#3498db",
        TaskState.OK: "#2ecc71",
        TaskState.DENIED: "#e74c3c",
        TaskState.PENDING: "#f39c12",
        TaskState.DISABLED: "#636e72",
    }

    @property
    def state(self) -> TaskState:
        return TaskState(self._node_data.get("_task_state", "SCHEDULED"))

    @state.setter
    def state(self, value: TaskState) -> None:
        self._node_data["_task_state"] = value.value
        self._node_data["color"] = self._COLORS[value]

    @property
    def text(self) -> str:
        return self._node_data.get("text", "")

    @text.setter
    def text(self, value: str) -> None:
        self._node_data["text"] = value

    @tool
    def get_text(self) -> str:
        """Get the current text of this task."""
        return self.text

    @tool
    def set_text(self, value: str) -> None:
        """Set the text of this task."""
        self.text = value

    @property
    def metadata(self) -> dict:
        return self._node_data

    def is_ready(self) -> bool:
        """True if all incoming edges are resolved and all enabled edges have OK predecessors."""
        incoming_edges = self._incoming_edges
        if not incoming_edges:
            return False

        # All incoming edges must be in terminal states
        if any(e.state == EdgeState.SCHEDULED for e in incoming_edges):
            return False

        # All enabled edges must point to OK predecessors
        for edge in incoming_edges:
            if edge.state == EdgeState.ENABLED:
                predecessor = edge.get_from_task()
                if predecessor.state != TaskState.OK:
                    return False

        return True

    @tool
    def accept(self) -> None:
        """Mark this task as completed successfully."""
        self.state = TaskState.OK

    @tool
    def deny(self) -> None:
        """Deny the task and reject the overall process."""
        self.state = TaskState.DENIED

    async def proceed(self) -> TaskState:
        """Invoke the agent on this task, evaluate edge conditions, and return the new state."""
        thread_id = f"{self._process_id}-{self.task_id}" if self._process_id else None

        # 1. Invoke agent — it calls accept(), deny(), or set_text() tools
        await self.invoke_agent(
            prompt=self.text,
            persistent_thread_id=thread_id,
        )

        # 2. If OK, evaluate outgoing edge conditions
        new_state = self.state
        if new_state == TaskState.OK:
            for edge in self._outgoing_edges:
                edge_prompt = f"Is the condition '{edge.condition}' met? Reply with a single word: 'yes' or 'no'."
                result = await self.invoke_agent(
                    prompt=edge_prompt,
                    output_schema=str,
                    persistent_thread_id=thread_id,
                )
                is_met = result == "yes" if result else False
                edge.state = EdgeState.ENABLED if is_met else EdgeState.DISABLED

        return new_state