from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import EdgeState

if TYPE_CHECKING:
    from .task import Task


class Edge:
    """Wraps a JSON edge object."""

    def __init__(self, edge_data: dict) -> None:
        self._edge_data = edge_data
        self._from_task: Task | None = None
        self._to_task: Task | None = None

    @property
    def edge_data(self) -> dict:
        return self._edge_data

    @property
    def from_task_id(self) -> str:
        return self._edge_data["fromNode"]

    @property
    def to_task_id(self) -> str:
        return self._edge_data["toNode"]

    @property
    def condition(self) -> str:
        return self._edge_data.get("label", "")

    @property
    def state(self) -> "EdgeState":
        return EdgeState(self._edge_data.get("_edge_state", "SCHEDULED"))

    @state.setter
    def state(self, value: "EdgeState") -> None:
        self._edge_data["_edge_state"] = value.value

    def get_from_task(self) -> Task:
        return self._from_task

    def get_to_task(self) -> Task:
        return self._to_task