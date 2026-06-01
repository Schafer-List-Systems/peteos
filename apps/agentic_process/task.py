from __future__ import annotations

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

from ._types import EdgeState, TaskState


class Task(AgenticObjectBase):
    """Each Task is an OAP object that wraps a JSON node object."""

    def __init__(self, node_data: dict) -> None:
        super().__init__()
        self._node_data = node_data
        self._outgoing_edges: list[Edge] = []
        self._incoming_edges: list[Edge] = []

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

    @property
    def state(self) -> TaskState:
        return TaskState(self._node_data.get("_task_state", "SCHEDULED"))

    @state.setter
    def state(self, value: TaskState) -> None:
        self._node_data["_task_state"] = value.value

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